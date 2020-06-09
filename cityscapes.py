#!/usr/bin/env python3
"""Cityscapesの実験用コード。

https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes

Gated-SCNN (Extra Training Data: ×): 82.8%
  - train 800x800

"""
import functools
import pathlib

import albumentations as A
import numpy as np
import tensorflow as tf

import pytoolkit as tk

num_classes = 19 + 1  # クラス+背景
input_shape = (800, 800, 3)
batch_size = 1
data_dir = pathlib.Path("data/cityscapes")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    create_model().check()


@app.command(use_horovod=True)
def train():
    train_set, val_set = load_data()
    model = create_model()
    model.train(train_set, val_set)
    _evaluate(model, val_set)


@app.command(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = create_model().load()
    _evaluate(model, val_set)


def _evaluate(model, val_set):
    if tk.hvd.is_master():
        pred_val = model.predict_flow(val_set)
        evals = tk.evaluations.print_ss(flow_labels(val_set), pred_val, 0.5)
        tk.notifications.post_evals(evals)
    tk.hvd.barrier()


def load_data():
    return tk.datasets.load_cityscapes(data_dir)


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        nfold=1,
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        epochs=300,
        callbacks=[tk.callbacks.CosineAnnealing()],
        models_dir=models_dir,
        model_name_format="model.h5",
        skip_if_exists=False,
        on_batch_fn=_tta,
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
        tk.layers.GroupNormalization, gamma_regularizer=tf.keras.regularizers.l2(1e-4),
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

    inputs = x = tf.keras.layers.Input(input_shape)
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
    x = conv2d(128, kernel_size=2, strides=2)(x)  # 1/4
    x = bn()(x)
    x = blocks(128, 2, down=False)(x)
    d = x
    x = blocks(256, 4)(x)  # 1/8
    x = blocks(512, 4)(x)  # 1/16
    x = blocks(512, 4)(x)  # 1/32
    x = conv2d(128 * 8 * 8, kernel_size=1)(x)
    x = bn()(x)
    x = act()(x)
    x = tk.layers.SubpixelConv2D(scale=8)(x)  # 1/4
    x = conv2d(128)(x)
    x = bn()(x)
    d = bn()(conv2d(128)(d))
    x = tf.keras.layers.add([x, d])
    x = blocks(128, 3, down=False)(x)
    x = tf.keras.layers.Conv2D(
        num_classes * 4 * 4,
        kernel_size=1,
        padding="same",
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        use_bias=True,
        bias_initializer=tf.keras.initializers.constant(tk.math.logit(0.01)),
    )(x)
    x = tk.layers.SubpixelConv2D(scale=4, name="logits")(x)  # 1/1
    x = tf.keras.layers.Activation(activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    base_lr = 1e-3 * batch_size * tk.hvd.size()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_lr, momentum=0.9, nesterov=True
    )

    def loss(y_true, y_pred):
        del y_pred
        logits = model.get_layer("logits").output
        losses = [
            tk.losses.lovasz_hinge(
                y_true[:, :, :, i], logits[:, :, :, i], from_logits=True
            )
            for i in range(num_classes)
        ]
        return tf.math.reduce_mean(losses, axis=0)

    tk.models.compile(
        model, optimizer, loss, [tk.metrics.binary_accuracy, tk.metrics.binary_iou]
    )
    return model, model


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, data_augmentation=False):
        super().__init__(batch_size=batch_size)
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug = A.Compose(
                [
                    tk.image.RandomTransform(size=input_shape[:2]),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
        else:
            self.aug = tk.image.Resize(size=input_shape[:2])

    def get_data(self, dataset: tk.data.Dataset, index: int):
        assert isinstance(dataset.metadata, dict)
        X, y = dataset.get_data(index)
        X = tk.ndimage.load(X)
        y = tk.ndimage.load(y)
        aug = self.aug(image=X, mask=y)
        X = aug["image"]
        y = aug["mask"]
        y = tk.ndimage.mask_to_onehot(
            y, dataset.metadata["class_colors"], append_bg=True
        )
        X = tk.ndimage.preprocess_tf(X)
        return X, y


def _tta(model, X_batch):
    return np.mean(
        [
            model.predict_on_batch(X_batch),
            model.predict_on_batch(X_batch[:, :, ::-1, :])[:, :, ::-1, :],
        ],
        axis=0,
    )


def flow_labels(dataset: tk.data.Dataset):
    assert isinstance(dataset.metadata, dict)
    for y in dataset.labels:
        y = tk.ndimage.load(y)
        y = tk.ndimage.mask_to_onehot(
            y, dataset.metadata["class_colors"], append_bg=True
        )
        yield y


if __name__ == "__main__":
    app.run(default="train")
