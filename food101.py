#!/usr/bin/env python3
"""転移学習の練習用コード。(Food-101)"""
import pathlib

import albumentations as A
import tensorflow as tf

import pytoolkit as tk

num_classes = 101
train_shape = (299, 299, 3)
predict_shape = (299, 299, 3)
batch_size = 16
data_dir = pathlib.Path(f"data/food-101")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    model = create_model().check()
    train_set, val_set = load_data()
    model.models = [model.create_model_fn()]
    model.evaluate(train_set)
    model.evaluate(val_set, prefix="val_")


@app.command(use_horovod=True)
def train():
    train_set, val_set = load_data()

    inputs = x = tf.keras.layers.Input((None, None, 3))
    backbone = tk.applications.xception.xception(input_tensor=x)
    x = backbone.output
    x = tk.layers.GeM2D()(x)
    pretrain_model = tf.keras.models.Model(inputs, x)
    feats_train = tk.models.predict(
        pretrain_model, train_set, MyDataLoader(), use_horovod=True
    )
    feats_val = tk.models.predict(
        pretrain_model, val_set, MyDataLoader(), use_horovod=True
    )
    if tk.hvd.is_master():
        import sklearn.linear_model

        estimator = sklearn.linear_model.LogisticRegression(
            C=0.01, solver="lbfgs", multi_class="multinomial", n_jobs=-1
        )
        estimator.fit(feats_train, train_set.labels)
        tk.utils.dump(
            [estimator.coef_, estimator.intercept_], models_dir / "linear.pkl"
        )
        print("acc:    ", estimator.score(feats_train, train_set.labels))
        print("val_acc:", estimator.score(feats_val, val_set.labels))
    tk.hvd.barrier()

    model = create_model()
    evals = model.train(train_set, val_set)
    tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = create_model().load(models_dir)
    pred = model.predict(val_set)[0]
    if tk.hvd.is_master():
        tk.evaluations.print_classification_metrics(val_set.labels, pred)


def load_data():
    return tk.datasets.load_trainval_folders(data_dir, swap=True)


def create_model():
    return MyModel(
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        fit_params={"epochs": 30, "callbacks": [tk.callbacks.CosineAnnealing()]},
        models_dir=models_dir,
        model_name_format="model.h5",
        use_horovod=True,
    )


class MyModel(tk.pipeline.KerasModel):
    """KerasModel"""

    def create_network(self) -> tf.keras.models.Model:
        inputs = x = tf.keras.layers.Input((None, None, 3))
        backbone = tk.applications.xception.xception(input_tensor=x)
        x = backbone.output
        x = tk.layers.GeM2D()(x)
        x = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer="zeros",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="logits",
        )(x)
        x = tf.keras.layers.Activation(activation="softmax")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)

        coef, intercept = tk.utils.load(models_dir / "linear.pkl")
        model.get_layer("logits").set_weights([coef.T, intercept])

        return model

    def create_optimizer(self, mode: str) -> tk.models.OptimizerType:
        del mode
        base_lr = 1e-3 * batch_size * tk.hvd.size()
        optimizer = tf.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)
        return optimizer

    def create_loss(self, model: tf.keras.models.Model) -> tuple:
        def loss(y_true, y_pred):
            del y_pred
            logits = model.get_layer("logits").output
            return tk.losses.categorical_crossentropy(
                y_true, logits, from_logits=True, label_smoothing=0.2
            )

        metrics = ["acc"]
        return loss, metrics


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, data_augmentation=False):
        super().__init__(
            batch_size=batch_size,
            data_per_sample=2 if data_augmentation else 1,
            parallel=True,
        )
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        width=train_shape[1],
                        height=train_shape[0],
                        base_scale=predict_shape[0] / train_shape[0],
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        else:
            self.aug1 = tk.image.Resize(width=predict_shape[1], height=predict_shape[0])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = tk.ndimage.load(X)
        X = self.aug1(image=X)["image"]
        y = tf.keras.utils.to_categorical(y, num_classes)
        return X, y

    def get_sample(self, data: list) -> tuple:
        if self.data_augmentation:
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
            X = self.aug2(image=X)["image"]
        else:
            X, y = super().get_sample(data)
        X = tk.ndimage.preprocess_tf(X)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
