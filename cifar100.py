#!/usr/bin/env python3
"""CIFAR-100

val_loss: 2.239
val_acc:  0.843

"""
import functools
import pathlib

import albumentations as A
import tensorflow as tf

import pytoolkit as tk

num_classes = 100
input_shape = (32, 32, 3)
batch_size = 32
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    create_model().check()


@app.command(use_horovod=True)
def train():
    train_set, val_set = tk.datasets.load_cifar100()
    model = create_model()
    evals = model.train(train_set, val_set)
    tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate():
    _, val_set = tk.datasets.load_cifar100()
    model = create_model().load(models_dir)
    pred = model.predict(val_set)[0]
    if tk.hvd.is_master():
        tk.evaluations.print_classification_metrics(val_set.labels, pred)


def create_model():
    return MyModel(
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        fit_params={"epochs": 300, "callbacks": [tk.callbacks.CosineAnnealing()]},
        models_dir=models_dir,
        model_name_format="model.h5",
        skip_if_exists=False,
        use_horovod=True,
    )


class MyModel(tk.pipeline.KerasModel):
    """KerasModel"""

    def create_network(self) -> tf.keras.models.Model:
        conv2d = functools.partial(
            tf.keras.layers.Conv2D,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )
        bn = functools.partial(
            tf.keras.layers.BatchNormalization,
            gamma_regularizer=tf.keras.regularizers.l2(1e-4),
        )
        act = functools.partial(tf.keras.layers.Activation, "relu")

        def down(filters):
            def layers(x):
                x = conv2d(filters, kernel_size=4, strides=2)(x)
                x = bn()(x)
                return x

            return layers

        def blocks(filters, count):
            def layers(x):
                for _ in range(count):
                    sc = x
                    x = conv2d(filters)(x)
                    x = bn()(x)
                    x = act()(x)
                    x = conv2d(filters)(x)
                    # resblockのadd前だけgammaの初期値を0にする。 <https://arxiv.org/abs/1812.01187>
                    x = bn(gamma_initializer="zeros")(x)
                    x = tf.keras.layers.add([sc, x])
                x = bn()(x)
                x = act()(x)
                return x

            return layers

        inputs = x = tf.keras.layers.Input(input_shape)
        x = conv2d(128)(x)
        x = bn()(x)
        x = blocks(128, 8)(x)
        x = down(256)(x)
        x = blocks(256, 8)(x)
        x = down(512)(x)
        x = blocks(512, 8)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="logits",
        )(x)
        x = tf.keras.layers.Activation(activation="softmax")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
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
                    tk.image.RandomTransform(width=32, height=32),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        else:
            self.aug1 = A.Compose([])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = self.aug1(image=X)["image"]
        y = tf.keras.utils.to_categorical(y, num_classes)
        return X, y

    def get_sample(self, data: list) -> tuple:
        if self.data_augmentation:
            sample1, sample2 = data
            # sample = tk.ndimage.cut_mix(sample1, sample2)
            X, y = tk.ndimage.mixup(sample1, sample2, mode="uniform")
            X = self.aug2(image=X)["image"]
        else:
            X, y = super().get_sample(data)
        X = tk.ndimage.preprocess_tf(X)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
