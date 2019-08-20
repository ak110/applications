#!/usr/bin/env python3
"""TGS Salt Identification Challengeの実験用コード。

- 一番良かったモデル(ls_darknet53_coord_hcs): 0.871

[INFO ] val_loss: 0.489
[INFO ] val_acc:  0.966
[INFO ] val_iou:  0.499
[INFO ] score:     0.860
[INFO ] IoU mean:  0.845
[INFO ] Acc empty: 0.945

"""
import functools
import pathlib

import numpy as np

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
    train_dataset, val_dataset = load_data()
    model = create_model()
    tk.training.train(
        model,
        train_dataset,
        val_dataset,
        train_preprocessor=MyPreprocessor(data_augmentation=True),
        val_preprocessor=MyPreprocessor(),
        batch_size=batch_size,
        epochs=600,
        callbacks=[tk.callbacks.CosineAnnealing()],
        model_path=models_dir / "model.h5",
    )
    _evaluate(model, val_dataset)


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def validate(model=None):
    _, val_dataset = load_data()
    model = model or tk.models.load(models_dir / "model.h5")
    _evaluate(model, val_dataset)


def _evaluate(model, val_dataset):
    pred_val = tk.models.predict(
        model,
        val_dataset,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
    )
    if tk.hvd.is_master():
        # スコア表示
        score = compute_score(val_dataset.labels, pred_val, 0.5)
        logger.info(f"score:     {score:.3f}")
        # オレオレ指標
        print_metrics(val_dataset.labels > 127, pred_val > 0.5, print_fn=logger.info)
        # 閾値探索
        _, _ = tk.ml.search_threshold(
            val_dataset.labels,
            pred_val,
            np.linspace(0.3, 0.7, 41),
            compute_score,
            "maximize",
        )
    tk.hvd.barrier()


def load_data():
    import pandas as pd

    def _load_image(X):
        X = np.array(
            [tk.ndimage.load(p, grayscale=True) for p in tk.utils.tqdm(X, desc="load")]
        )
        return X

    id_list = pd.read_csv(data_dir / "train.csv")["id"].values
    X = _load_image([data_dir / "train" / "images" / f"{id_}.png" for id_ in id_list])
    y = _load_image([data_dir / "train" / "masks" / f"{id_}.png" for id_ in id_list])
    ti, vi = tk.ml.cv_indices(
        X, y, cv_count=5, cv_index=0, split_seed=6768115, stratify=False
    )
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])

    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_val, y_val)


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

    def down(filters):
        def layers(x):
            g = tk.keras.layers.Conv2D(
                1,
                3,
                padding="same",
                activation="sigmoid",
                kernel_regularizer=tk.keras.regularizers.l2(1e-4),
            )(x)
            x = tk.keras.layers.multiply([x, g])
            x = tk.keras.layers.MaxPooling2D(2, strides=1, padding="same")(x)
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
    x = tk.layers.Pad2D(((5, 6), (5, 6)), mode="reflect")(x)  # 112
    x = tk.keras.layers.concatenate([x, x, x])
    x = tk.layers.CoordChannel2D(x_channel=False)(x)
    x = conv2d(64, kernel_size=8)(x)
    x = bn()(x)
    x = act()(x)
    x = conv2d(128, kernel_size=2, strides=2)(x)  # 1/2
    x = bn()(x)
    x = blocks(128, 2)(x)
    x = down(256)(x)  # 1/4
    x = blocks(256, 4)(x)
    d = x
    x = down(512)(x)  # 1/8
    x = blocks(512, 4)(x)
    x = down(512)(x)  # 1/16
    x = blocks(512, 4)(x)
    x = conv2d(128 * 4 * 4)(x)
    x = bn()(x)
    x = act()(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 1/4
    x = conv2d(256)(x)
    x = bn()(x)
    d = bn()(conv2d(256)(d))
    x = tk.keras.layers.add([x, d])
    x = blocks(256, 3)(x)
    x = conv2d(
        1 * 4 * 4,
        use_bias=True,
        bias_initializer=tk.keras.initializers.constant(tk.math.logit(0.01)),
    )(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 1/1
    x = tk.keras.layers.Cropping2D(((5, 6), (5, 6)))(x)  # 101
    logits = x
    x = tk.keras.layers.Activation("sigmoid")(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)

    def loss(y_true, y_pred):
        _ = y_pred
        return tk.losses.lovasz_hinge(y_true, logits, from_logits=True)

    base_lr = 1e-3 * batch_size * tk.hvd.size()
    optimizer = tk.keras.optimizers.SGD(
        lr=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0
    )
    tk.models.compile(
        model, optimizer, loss, [tk.metrics.binary_accuracy, tk.metrics.binary_iou]
    )
    return model


def compute_score(y_true, y_pred, threshold):
    """スコア算出。"""
    y_true = np.int32(y_true > 127)
    y_pred = np.int32(y_pred > threshold)

    obj = np.any(y_true, axis=(1, 2, 3))
    empty = np.logical_not(obj)
    pred_empty = np.logical_not(np.any(y_pred, axis=(1, 2, 3)))
    tn = np.logical_and(empty, pred_empty)

    inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)

    prec_list = []
    for th in np.arange(0.5, 1.0, 0.05):
        pred_obj = iou > th
        match = np.logical_and(obj, pred_obj) + tn
        prec_list.append(np.sum(match) / len(y_true))
    return np.mean(prec_list)


def print_metrics(y_true, y_pred, print_fn):
    """オレオレ指標。"""
    obj = np.any(y_true, axis=(1, 2, 3))
    empty = np.logical_not(obj)

    # 答えが空でないときのIoUの平均
    inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)
    iou_mean = np.mean(iou[obj])
    print_fn(f"IoU mean:  {iou_mean:.3f}")

    # 答えが空の場合の正解率
    pred_empty = np.logical_not(np.any(y_pred, axis=(1, 2, 3)))
    acc_empty = np.sum(np.logical_and(empty, pred_empty)) / np.sum(empty)
    print_fn(f"Acc empty: {acc_empty:.3f}")


class MyPreprocessor(tk.data.Preprocessor):
    """Preprocessor。"""

    def __init__(self, data_augmentation=False):
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug = tk.image.Compose(
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
        d = self.aug(image=X, mask=y)
        X = tk.ndimage.preprocess_tf(d["image"])
        y = d["mask"] / 255
        return X, y


if __name__ == "__main__":
    app.run(default="train")
