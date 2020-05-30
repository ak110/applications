#!/usr/bin/env python3
"""imagenetteの実験用コード。(trainとvalをひっくり返している。)

val_loss: 1.883
val_acc:  0.879

"""
import functools
import pathlib
import typing

import albumentations as A
import tensorflow as tf

import pytoolkit as tk

params: typing.Dict[str, typing.Any] = {
    "num_classes": 10,
    "train_shape": (320, 320, 3),
    "predict_shape": (480, 480, 3),
    "base_lr": 1e-3,
    "batch_size": 16,
    "epochs": 1800,
}
data_dir = pathlib.Path("data/imagenette")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir, use_horovod=True)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    checking_set = load_data()[0].slice(range(16))
    training_model, prediction_model = create_network(len(checking_set))
    tk.models.check(
        training_model=training_model,
        prediction_model=prediction_model,
        models_dir=models_dir,
        dataset=checking_set,
        training_data_loader=MyDataLoader(mode="training"),
        prediction_data_loader=MyDataLoader(mode="prediction"),
        save_mode="hdf5",
    )


@app.command()
def train():
    training_set, validation_set = load_data()
    training_model, prediction_model = create_network(len(training_set))
    tk.models.fit(
        training_model,
        training_iterator=MyDataLoader(mode="training").load(training_set),
        validation_iterator=MyDataLoader(mode="prediction").load(validation_set),
        epochs=params["epochs"],
    )
    tk.models.save(prediction_model, models_dir / "model.h5")
    if params["refine_epochs"] > 0:
        tk.models.freeze_layers(training_model, tf.keras.layers.BatchNormalization)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=params["base_lr"] * app.num_workers,
            momentum=0.9,
            nesterov=True,
        )
        tk.models.compile(training_model, optimizer, loss, ["acc"])
        tk.models.fit(
            training_model,
            training_iterator=MyDataLoader(mode="refining").load(training_set),
            validation_iterator=MyDataLoader(mode="prediction").load(validation_set),
            epochs=params["refine_epochs"],
        )
        tk.models.save(prediction_model, models_dir / "model.h5")
    validate(prediction_model)


@app.command()
def validate(prediction_model=None):
    training_set, validation_set = load_data()
    prediction_model = prediction_model or create_network(len(training_set))[1]
    tk.models.load_weights(prediction_model, models_dir / "model.h5")
    pred = tk.models.predict(
        prediction_model, MyDataLoader(mode="predict").load(validation_set)
    )
    tk.evaluations.print_classification_metrics(validation_set.labels, pred)


def load_data():
    return tk.datasets.load_trainval_folders(data_dir, swap=True)


def create_network(train_size):
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

    def blocks(filters, count, down=True):
        def layers(x):
            if down:
                in_filters = x.shape[-1]
                g = conv2d(in_filters)(x)
                g = bn()(g)
                g = act()(g)
                g = conv2d(in_filters, use_bias=True, activation="sigmoid")(g)
                x = tf.keras.layers.multiply([x, g])
                x = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
                x = tk.layers.BlurPooling2D(taps=4)(x)
                x = conv2d(filters)(x)
                x = bn()(x)
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

    inputs = x = tf.keras.layers.Input((None, None, 3))
    x = tf.keras.layers.concatenate(
        [
            conv2d(16, kernel_size=2, strides=2)(x),
            conv2d(16, kernel_size=4, strides=2)(x),
            conv2d(16, kernel_size=6, strides=2)(x),
            conv2d(16, kernel_size=8, strides=2)(x),
        ]
    )  # 1/2
    x = bn()(x)
    x = act()(x)
    x = blocks(128, 3)(x)  # 1/4
    x = blocks(256, 3)(x)  # 1/8
    x = blocks(512, 3)(x)  # 1/16
    x = blocks(512, 3)(x)  # 1/32
    x = tk.layers.GeMPooling2D()(x)
    x = tf.keras.layers.Dense(
        params["num_classes"],
        kernel_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    global_batch_size = params["batch_size"] * app.num_workers
    schedule = tk.schedules.CosineAnnealing(
        initial_learning_rate=params["base_lr"] * global_batch_size,
        decay_steps=-(-train_size // global_batch_size) * params["epochs"],
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=schedule, momentum=0.9, nesterov=True
    )

    tk.models.compile(model, optimizer, loss, ["acc"])

    x = tf.keras.layers.Activation("softmax")(x)
    prediction_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model, prediction_model


def loss(y_true, logits):
    return tk.losses.categorical_crossentropy(
        y_true, logits, from_logits=True, label_smoothing=0.2
    )


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, mode):
        super().__init__(batch_size=params["batch_size"])
        self.mode = mode
        self.aug2: typing.Any = None
        if self.mode == "training":
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        size=params["train_shape"][:2],
                        base_scale=params["predict_shape"][0]
                        / params["train_shape"][0],
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
            self.data_per_sample = 2  # mixup
        elif self.mode == "refining":
            self.aug1 = tk.image.RandomTransform.create_refine(
                size=params["predict_shape"][:2]
            )
            self.aug2 = None
        else:
            self.aug1 = tk.image.Resize(size=params["predict_shape"][:2])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = tk.ndimage.load(X)
        X = self.aug1(image=X)["image"]
        y = (
            tf.keras.utils.to_categorical(y, params["num_classes"])
            if y is not None
            else None
        )
        return X, y

    def get_sample(self, data):
        if len(data) == 2:
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
            X = self.aug2(image=X)["image"]
        else:
            X, y = super().get_sample(data)
        X = tk.ndimage.preprocess_tf(X)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
