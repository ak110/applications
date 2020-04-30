#!/usr/bin/env python3
"""TGS Salt Identification Challengeの実験用コード。

Private: 0.87401
Public:  0.85624

iou:       [0.96  0.875]
miou:      0.917
iou_score: 0.871
dice:      0.521
fg_iou:    0.851
bg_acc:    0.954
acc:       0.969

"""
import functools
import pathlib

import albumentations as A
import numpy as np
import pandas as pd
import tensorflow as tf

import pytoolkit as tk

input_shape = (101, 101, 1)
batch_size = 16
data_dir = pathlib.Path("data/kaggle_salt")
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
    _predict(model)


@app.command(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = create_model().load()
    _evaluate(model, val_set)
    _predict(model)


@app.command(use_horovod=True)
def predict():
    model = create_model().load()
    _predict(model)


def _evaluate(model, val_set):
    pred_val = model.predict(val_set, fold=0)
    if tk.hvd.is_master():
        evals = tk.evaluations.print_ss_metrics(val_set.labels / 255, pred_val, 0.5)
        tk.notifications.post_evals(evals)
    tk.hvd.barrier()


def _predict(model):
    test_set = load_test_data()
    test_set.labels = np.zeros_like(test_set.data)  # エラー除けのダミー
    pred_test = model.predict(test_set, fold=0)

    if tk.hvd.is_master():
        df = pd.DataFrame()
        df["id"] = test_set.ids
        df["rle_mask"] = tk.utils.encode_rl_array(pred_test >= 0.5)
        df.to_csv(str(models_dir / "submission.csv"), index=False)
    tk.hvd.barrier()


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
    x = tf.pad(x, ((0, 0), (5, 6), (5, 6), (0, 0)), mode="reflect")  # 112
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.concatenate([x, x, x])
    backbone = tk.applications.darknet53.create(input_tensor=x)
    x = backbone.output
    # x = tk.layers.ScaleGradient(scale=0.1)(x)
    x = conv2d(256)(x)
    x = bn()(x)
    x = act()(x)
    x = conv2d(32 * 8 * 8, kernel_size=1)(x)
    x = bn()(x)
    x = act()(x)
    x = tk.layers.SubpixelConv2D(scale=8)(x)  # 1/4
    x = tk.layers.CoordChannel2D(x_channel=False)(x)
    x = conv2d(256)(x)
    x = bn()(x)
    d = tk.applications.darknet53.get_1_over_4(backbone)  # 1/4
    # d = tk.layers.ScaleGradient(scale=0.1)(d)
    d = conv2d(256)(d)
    d = bn(center=False)(d)
    x = tf.keras.layers.add([x, d])
    x = blocks(256, 8)(x)
    x = conv2d(
        1 * 2 * 2,
        use_bias=True,
        bias_initializer=tf.keras.initializers.constant(tk.math.logit(0.01)),
    )(x)
    x = tk.layers.SubpixelConv2D(scale=2)(x)  # 1/1
    x = tf.keras.layers.Cropping2D(((5, 6), (5, 6)))(x)  # 101
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    learning_rate = 1e-3 * batch_size * tk.hvd.size() * app.num_replicas_in_sync
    optimizer = tk.optimizers.SGDEx(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
        lr_multipliers={backbone: 0.1},
    )

    def loss(y_true, logits):
        return tk.losses.lovasz_hinge(y_true, logits, from_logits=True)

    tk.models.compile(
        model, optimizer, loss, [tk.metrics.binary_accuracy, tk.metrics.binary_iou]
    )

    x = tf.keras.layers.Activation("sigmoid")(x)
    prediction_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model, prediction_model


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, data_augmentation=False):
        super().__init__(
            batch_size=batch_size, data_per_sample=2 if data_augmentation else 1
        )
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug = A.Compose(
                [
                    tk.image.RandomTransform(size=input_shape[:2]),
                    tk.image.RandomBlur(p=0.125),
                    tk.image.RandomUnsharpMask(p=0.125),
                    tk.image.RandomBrightness(p=0.25),
                    tk.image.RandomContrast(p=0.25),
                ]
            )
        else:
            self.aug = tk.image.Resize(size=input_shape[:2])

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        d = self.aug(image=X, mask=y)
        X = tk.applications.darknet53.preprocess_input(d["image"])
        y = d["mask"] / 255
        y = y.reshape(input_shape)
        return X, y

    def get_sample(self, data):
        if self.data_augmentation:
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
        else:
            X, y = super().get_sample(data)
        return X, y


def _tta(model, X_batch):
    return np.mean(
        [
            model.predict_on_batch(X_batch),
            model.predict_on_batch(X_batch[:, :, ::-1, :])[:, :, ::-1, :],
        ],
        axis=0,
    )


if __name__ == "__main__":
    app.run(default="train")
