#!/usr/bin/env python3
"""imagenetteの実験用コード。

Weight Standalization + GroupNormalization。重いので普段は使わない。

val_loss: 1.826
val_acc:  0.855

"""
import functools
import pathlib
import typing

import albumentations as A
import tensorflow as tf
import tensorflow_addons as tfa

import pytoolkit as tk

num_classes = 10
train_shape = (320, 320, 3)
predict_shape = (480, 480, 3)
batch_size = 16
data_dir = pathlib.Path("data/imagenette")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    create_model().check(load_data()[0].slice(range(16)))


@app.command(use_horovod=True)
def train():
    train_set, val_set = load_data()
    model = create_model()
    evals = model.train(train_set, val_set)
    tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate(model=None):
    _, val_set = load_data()
    model = create_model().load()
    pred = model.predict(val_set, fold=0)
    if tk.hvd.is_master():
        tk.evaluations.print_classification(val_set.labels, pred)


def load_data():
    return tk.datasets.load_trainval_folders(data_dir, swap=True)


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        nfold=1,
        models_dir=models_dir,
        train_data_loader=MyDataLoader(mode="train"),
        refine_data_loader=MyDataLoader(mode="refine"),
        val_data_loader=MyDataLoader(mode="test"),
        epochs=1800,
        callbacks=[tk.callbacks.CosineAnnealing()],
        model_name_format="model.h5",
        skip_if_exists=False,
    )


def create_network():
    conv2d = functools.partial(
        tk.layers.WSConv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tfa.layers.GroupNormalization,
        groups=32,
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
                g = tf.keras.layers.Conv2D(
                    in_filters,
                    3,
                    padding="same",
                    kernel_initializer="he_uniform",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                    use_bias=True,
                    activation="sigmoid",
                )(g)
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
    x = conv2d(64, kernel_size=6, strides=2)(x)  # 1/2
    x = bn()(x)
    x = act()(x)
    x = blocks(128, 4)(x)  # 1/4
    x = blocks(256, 4)(x)  # 1/8
    x = blocks(512, 4)(x)  # 1/16
    x = blocks(512, 4)(x)  # 1/32
    x = tk.layers.GeMPooling2D()(x)
    x = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer="zeros",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    learning_rate = 1e-3 * batch_size * tk.hvd.size() * app.num_replicas_in_sync
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True
    )

    def loss(y_true, logits):
        return tk.losses.categorical_crossentropy(
            y_true, logits, from_logits=True, label_smoothing=0.2
        )

    tk.models.compile(model, optimizer, loss, ["acc"])

    x = tf.keras.layers.Activation("softmax")(x)
    pred_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model, pred_model


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, mode):
        super().__init__(
            batch_size=batch_size, data_per_sample=2 if mode == "train" else 1
        )
        self.mode = mode
        self.aug2: typing.Any = None
        if self.mode == "train":
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
        elif self.mode == "refine":
            self.aug1 = tk.image.RandomTransform.create_refine(size=predict_shape[:2])
            self.aug2 = None
        else:
            self.aug1 = tk.image.Resize(size=predict_shape[:2])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = tk.ndimage.load(X)
        X = self.aug1(image=X)["image"]
        y = tf.keras.utils.to_categorical(y, num_classes) if y is not None else None
        return X, y

    def get_sample(self, data):
        if self.mode == "train":
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
            X = self.aug2(image=X)["image"]
        else:
            X, y = super().get_sample(data)
        X = tk.ndimage.preprocess_tf(X)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
