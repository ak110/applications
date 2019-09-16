#!/usr/bin/env python3
"""TGS Salt Identification Challengeの実験用コード。

[INFO ] IoU:            [0.957 0.868]
[INFO ] mIoU:           0.912
[INFO ] IoU score:      0.868
[INFO ] Dice coef.:     0.524
[INFO ] IoU mean:       0.855
[INFO ] Acc empty:      0.942
[INFO ] Pixel Accuracy: 0.967

"""
import functools
import pathlib

import albumentations as A
import numpy as np
import pandas as pd

import pytoolkit as tk

input_shape = (101, 101, 1)
batch_size = 16
data_dir = pathlib.Path(f"data/kaggle_salt")
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
        train_preprocessor=MyPreprocessor(data_augmentation=True),
        val_preprocessor=MyPreprocessor(),
        batch_size=batch_size,
        epochs=300,
        callbacks=[tk.callbacks.CosineAnnealing()],
        model_path=models_dir / "model.h5",
    )
    _evaluate(model, val_set)
    _predict(model)


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = tk.models.load(models_dir / "model.h5")
    _evaluate(model, val_set)


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def predict():
    model = tk.models.load(models_dir / "model.h5")
    _predict(model)


def _evaluate(model, val_set):
    pred_val = tk.models.predict(
        model,
        val_set,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
        on_batch_fn=_tta,
    )
    if tk.hvd.is_master():
        tk.evaluations.print_ss_metrics(val_set.labels / 255, pred_val, 0.5)
    tk.hvd.barrier()


def _predict(model):
    test_set = load_test_data()
    test_set.labels = np.zeros_like(test_set.data)  # エラー除けのダミー
    pred_test = tk.models.predict(
        model,
        test_set,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
        on_batch_fn=_tta,
    )

    if tk.hvd.is_master():
        df = pd.DataFrame()
        df["id"] = test_set.ids
        df["rle_mask"] = tk.utils.encode_rl_array(pred_test >= 0.5)
        df.to_csv(str(models_dir / "submission.csv"), index=False)
    tk.hvd.barrier()


def _tta(model, X_batch):
    return np.mean(
        [
            model.predict_on_batch(X_batch),
            model.predict_on_batch(X_batch[:, :, ::-1, :])[:, :, ::-1, :],
        ],
        axis=0,
    )


def load_data():
    id_list = pd.read_csv(data_dir / "train.csv")["id"].values
    X = _load_image([data_dir / "train" / "images" / f"{id_}.png" for id_ in id_list])
    y = _load_image([data_dir / "train" / "masks" / f"{id_}.png" for id_ in id_list])
    ti, vi = tk.ml.cv_indices(
        X, y, cv_count=5, cv_index=0, split_seed=6768115, stratify=False
    )
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])
    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_val, y_val)


def load_test_data():
    id_list = pd.read_csv(data_dir / "sample_submission.csv")["id"].values
    X = _load_image([data_dir / "test" / "images" / f"{id_}.png" for id_ in id_list])
    return tk.data.Dataset(X, ids=id_list)


def _load_image(X):
    X = np.array(
        [tk.ndimage.load(p, grayscale=True) for p in tk.utils.tqdm(X, desc="load")]
    )
    return X


def create_model():
    conv2d = functools.partial(
        tk.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tk.keras.layers.BatchNormalization,
        gamma_regularizer=tk.keras.regularizers.l2(1e-4),
    )
    act = functools.partial(tk.keras.layers.Activation, "relu")

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
    x = tk.layers.Pad2D(((5, 6), (5, 6)), mode="reflect")(x)  # 112
    x = tk.keras.layers.concatenate([x, x, x])
    backbone = tk.applications.darknet53.darknet53(input_tensor=x, for_small=True)
    x = backbone.output
    x = tk.layers.ScaleGradient(scale=0.1)(x)
    x = conv2d(256)(x)
    x = bn()(x)
    x = act()(x)
    x = conv2d(256 * 4 * 4)(x)
    x = bn()(x)
    x = act()(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 1/4

    x = tk.layers.CoordChannel2D(x_channel=False)(x)
    x = conv2d(256)(x)
    x = bn()(x)
    d = backbone.get_layer("block12_add").output  # 1/4
    d = tk.layers.ScaleGradient(scale=0.1)(d)
    d = conv2d(256)(d)
    d = bn()(d)
    x = tk.keras.layers.add([x, d])
    x = blocks(256, 3)(x)
    x = conv2d(32 * 4 * 4)(x)
    x = bn()(x)
    x = act()(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 1/1

    x = tk.layers.CoordChannel2D(x_channel=False)(x)
    x = conv2d(32)(x)
    x = bn()(x)
    d = backbone.get_layer("block2_add").output  # 1/1
    d = tk.layers.ScaleGradient(scale=0.1)(d)
    d = conv2d(32)(d)
    d = bn()(d)
    x = tk.keras.layers.add([x, d])
    x = blocks(32, 3)(x)

    edge_logits = tk.keras.layers.Cropping2D(((5, 6), (5, 6)))(
        conv2d(
            1,
            use_bias=True,
            bias_initializer=tk.keras.initializers.constant(tk.math.logit(0.01)),
        )(x)
    )

    x = conv2d(
        1,
        use_bias=True,
        bias_initializer=tk.keras.initializers.constant(tk.math.logit(0.01)),
    )(x)
    x = tk.keras.layers.Cropping2D(((5, 6), (5, 6)))(x)  # 101
    logits = x

    x = tk.keras.layers.Activation("sigmoid")(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)

    def loss(y_true, y_pred):
        del y_pred

        loss_main = tk.losses.lovasz_hinge(y_true, logits, from_logits=True)

        tf = tk.tf
        edge_true = tf.image.sobel_edges(y_true)
        edge_true = tf.math.sqrt(edge_true[:, :, :, :, 0] * edge_true[:, :, :, :, 1])
        edge_true = tf.math.maximum(edge_true, 0)
        loss_edge = tk.losses.lovasz_hinge(edge_true, edge_logits, from_logits=True)

        return loss_main * 0.75 + loss_edge * 0.25

    base_lr = 1e-3 * batch_size * tk.hvd.size()
    optimizer = tk.keras.optimizers.SGD(
        lr=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0
    )
    tk.models.compile(
        model, optimizer, loss, [tk.metrics.binary_accuracy, tk.metrics.binary_iou]
    )
    return model


class MyPreprocessor(tk.data.Preprocessor):
    """Preprocessor。"""

    def __init__(self, data_augmentation=False):
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug = A.Compose(
                [
                    tk.image.RandomTransform(
                        width=input_shape[1], height=input_shape[0]
                    ),
                    tk.image.RandomBlur(p=0.125),
                    tk.image.RandomUnsharpMask(p=0.125),
                    tk.image.RandomBrightness(p=0.25),
                    tk.image.RandomContrast(p=0.25),
                ]
            )
        else:
            self.aug = tk.image.Resize(width=input_shape[1], height=input_shape[0])

    def get_sample(
        self, dataset: tk.data.Dataset, index: int, random: np.random.RandomState
    ):
        X, y = dataset.get_sample(index)
        d = self.aug(image=X, mask=y, rand=random)
        X = tk.applications.darknet53.preprocess_input(d["image"])
        y = d["mask"] / 255
        y = y.reshape(input_shape)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
