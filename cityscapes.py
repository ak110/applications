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

import pytoolkit as tk

num_classes = 19 + 1  # クラス+背景
input_shape = (800, 800, 3)
batch_size = 1
data_dir = pathlib.Path(f"data/cityscapes")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
@tk.dl.wrap_session()
def check():
    model = create_model()
    tk.training.check(model, plot_path=models_dir / "model.svg")


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def train():
    train_set, val_set = load_data()
    model = create_model()
    tk.training.train(
        model,
        train_set=train_set,
        val_set=val_set,
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        epochs=300,
        callbacks=[tk.callbacks.CosineAnnealing()],
        model_path=models_dir / "model.h5",
    )
    _evaluate(model, val_set)


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = tk.models.load(models_dir / "model.h5")
    _evaluate(model, val_set)


def _evaluate(model, val_set):
    if tk.hvd.is_master():
        pred_val = tk.models.predict(
            model,
            val_set,
            MyDataLoader(),
            batch_size=batch_size * 2,
            flow=True,
            on_batch_fn=_tta,
        )
        evals = tk.evaluations.print_ss_metrics(flow_labels(val_set), pred_val, 0.5)
        tk.notifications.post_evals(evals)
    tk.hvd.barrier()


def load_data():
    return tk.datasets.load_cityscapes(data_dir)


def _tta(model, X_batch):
    return np.mean(
        [
            model.predict_on_batch(X_batch),
            model.predict_on_batch(X_batch[:, :, ::-1, :])[:, :, ::-1, :],
        ],
        axis=0,
    )


def create_model():
    conv2d = functools.partial(tk.layers.WSConv2D, kernel_size=3)
    bn = functools.partial(
        tk.layers.GroupNormalization, gamma_regularizer=tk.keras.regularizers.l2(1e-4)
    )
    act = functools.partial(tk.keras.layers.Activation, "relu")

    def down(filters):
        def layers(x):
            in_filters = tk.K.int_shape(x)[-1]
            g = conv2d(in_filters // 8)(x)
            g = bn()(g)
            g = act()(g)
            g = tk.keras.layers.Conv2D(
                in_filters,
                3,
                padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=tk.keras.regularizers.l2(1e-4),
                use_bias=True,
                activation="sigmoid",
            )(g)
            x = tk.keras.layers.multiply([x, g])
            x = tk.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
            x = tk.layers.BlurPooling2D(taps=4)(x)
            x = conv2d(filters)(x)
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
                x = tk.keras.layers.add([sc, x])
            x = bn()(x)
            x = act()(x)
            return x

        return layers

    inputs = x = tk.keras.layers.Input(input_shape)
    x = tk.keras.layers.concatenate(
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
    x = blocks(128, 2)(x)
    d = x
    x = down(256)(x)  # 1/8
    x = blocks(256, 4)(x)
    x = down(512)(x)  # 1/16
    x = blocks(512, 4)(x)
    x = down(512)(x)  # 1/32
    x = blocks(512, 4)(x)
    x = conv2d(128 * 8 * 8, kernel_size=1)(x)
    x = bn()(x)
    x = act()(x)
    x = tk.layers.SubpixelConv2D(scale=8)(x)  # 1/4
    x = conv2d(128)(x)
    x = bn()(x)
    d = bn()(conv2d(128)(d))
    x = tk.keras.layers.add([x, d])
    x = blocks(128, 3)(x)
    x = tk.keras.layers.Conv2D(
        num_classes * 4 * 4,
        kernel_size=1,
        padding="same",
        kernel_initializer="he_uniform",
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
        use_bias=True,
        bias_initializer=tk.keras.initializers.constant(tk.math.logit(0.01)),
    )(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 1/1
    logits = x
    x = tk.keras.layers.Activation(activation="softmax")(logits)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 1e-3 * batch_size * tk.hvd.size()
    optimizer = tk.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)

    def loss(y_true, y_pred):
        del y_pred
        tf = tk.tf
        losses = [
            tk.losses.lovasz_hinge(
                y_true[:, :, :, i], logits[:, :, :, i], from_logits=True
            )
            for i in range(num_classes)
        ]
        return tf.reduce_mean(losses, axis=0)

    tk.models.compile(
        model, optimizer, loss, [tk.metrics.binary_accuracy, tk.metrics.binary_iou]
    )
    return model


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, data_augmentation=False):
        super().__init__(batch_size=batch_size, parallel=True)
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug = A.Compose(
                [
                    tk.image.RandomTransform(
                        width=input_shape[1], height=input_shape[0]
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
        else:
            self.aug = tk.image.Resize(width=input_shape[1], height=input_shape[0])

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_sample(index)
        X = tk.ndimage.load(X)
        y = tk.ndimage.load(y)
        y = tk.ndimage.mask_to_onehot(
            y, dataset.metadata["class_colors"], append_bg=True  # type: ignore
        )
        aug = self.aug(image=X, masks=[y[:, :, i] for i in range(y.shape[-1])])
        X = aug["image"]
        y = aug["masks"]
        y = np.concatenate(y, axis=2)
        X = tk.ndimage.preprocess_tf(X)
        y = y / 255
        return X, y


def flow_labels(dataset):
    for y in dataset.labels:
        y = tk.ndimage.load(y)
        y = tk.ndimage.mask_to_onehot(
            y, dataset.metadata["class_colors"], append_bg=True
        )
        y = y / 255
        yield y


if __name__ == "__main__":
    app.run(default="train")
