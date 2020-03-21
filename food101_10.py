#!/usr/bin/env python3
"""転移学習の練習用コード。(Food-101)

train: 25250 -> 1010 samples
val: 75750 samples

val_acc:     0.523

"""
import pathlib

import albumentations as A
import tensorflow as tf

import pytoolkit as tk

num_classes = 101
train_shape = (256, 256, 3)
predict_shape = (256, 256, 3)
batch_size = 16
data_dir = pathlib.Path(f"data/food-101")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    create_model().check(load_data()[0].slice(list(range(10))))


@app.command(use_horovod=True)
def train():
    train_set, val_set = load_data()
    model = create_model()
    evals = model.train(train_set, val_set)
    tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = create_model().load()
    pred = model.predict(val_set, fold=0)
    if tk.hvd.is_master():
        tk.evaluations.print_classification_metrics(val_set.labels, pred)


def load_data():
    train_set, val_set = tk.datasets.load_trainval_folders(data_dir, swap=True)
    train_set = tk.datasets.extract_class_balanced(train_set, num_classes, 10)
    return train_set, val_set


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        score_fn=tk.evaluations.evaluate_classification,
        nfold=1,
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        epochs=1800,
        callbacks=[tk.callbacks.CosineAnnealing(factor=0.1)],
        models_dir=models_dir,
        model_name_format="model.h5",
        skip_if_exists=False,
    )


def create_network():
    inputs = x = tf.keras.layers.Input((None, None, 3))
    backbone = tk.applications.efficientnet.create_b3(input_tensor=x)
    x = backbone.output
    x = tk.layers.GeMPooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, kernel_initializer="zeros")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    base_lr = 1e-4 * batch_size * tk.hvd.size()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_lr, momentum=0.9, nesterov=True
    )

    def loss(y_true, logits):
        return tk.losses.categorical_crossentropy(
            y_true, logits, from_logits=True, label_smoothing=0.2
        )

    tk.models.compile(model, optimizer, loss, ["acc"])

    x = tf.keras.layers.Activation(activation="softmax")(x)
    prediction_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    tk.models.compile(prediction_model, optimizer, loss, ["acc"])
    return model, prediction_model


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, data_augmentation=False):
        super().__init__(
            batch_size=batch_size, data_per_sample=2 if data_augmentation else 1,
        )
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        size=train_shape[:2],
                        base_scale=predict_shape[0] / train_shape[0],
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        else:
            self.aug1 = tk.image.Resize(size=predict_shape[:2])
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
