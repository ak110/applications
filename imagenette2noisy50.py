#!/usr/bin/env python3
"""Imagenette w/Label Noise = 50%の実験用コード。

<https://github.com/fastai/imagenette#imagenette-wlabel-noise--50>

## レギュレーション

- No inference time tricks, e.g. no: TTA, validation size > train size
- Must start with random weights
- Must be one of the size/#epoch combinations listed in the table

## 実行結果 (LB: 128px/20epochs, 79.36%)

acc:     0.756
error:   0.244
f1:      0.751
auc:     0.966
ap:      0.857
prec:    0.783
rec:     0.756
mcc:     0.733
logloss: 1.184

"""
from __future__ import annotations

import functools
import pathlib
import typing

import albumentations as A
import numpy as np
import pandas as pd
import tensorflow as tf

import pytoolkit as tk

params: dict[str, typing.Any] = {
    "num_classes": 10,
    "train_shape": (256, 256, 3),
    "predict_shape": (256, 256, 3),
    "base_lr": 1e-3,
    "batch_size": 16,
    "epochs": 80,
    "refine_epochs": 0,
}
data_dir = pathlib.Path("data/imagenette2")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
# app = tk.cli.App(output_dir=models_dir, use_horovod=True)
app = tk.cli.App(
    output_dir=models_dir, distribute_strategy_fn=tf.distribute.MirroredStrategy
)
logger = tk.log.get(__name__)


# tf.keras.mixed_precision.set_global_policy("mixed_float16")


@app.command(logfile=False)
def check():
    check_set = load_data()[0].slice(range(16))
    train_model, pred_model = create_network(len(check_set))
    tk.models.check(
        train_model=train_model,
        pred_model=pred_model,
        models_dir=models_dir,
        dataset=check_set,
        train_data_loader=MyDataLoader(mode="train"),
        pred_data_loader=MyDataLoader(mode="pred"),
        save_mode="hdf5",
    )


@app.command()
def train():
    train_set, val_set = load_data()
    train_model, pred_model = create_network(len(train_set))
    tk.models.fit(
        train_model,
        train_iterator=MyDataLoader(mode="train").load(train_set),
        val_iterator=MyDataLoader(mode="pred").load(val_set),
        epochs=params["epochs"],
    )
    tk.models.save(pred_model, models_dir / "model.h5")
    if params["refine_epochs"] > 0:
        tk.models.freeze_layers(train_model, tf.keras.layers.BatchNormalization)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=params["base_lr"] / 100 * app.num_workers,
            momentum=0.9,
            nesterov=True,
        )
        tk.models.compile(
            train_model,
            optimizer,
            loss,
            ["acc", tk.metrics.CosineSimilarity(from_logits=True)],
        )
        tk.models.fit(
            train_model,
            train_iterator=MyDataLoader(mode="refine").load(train_set),
            val_iterator=MyDataLoader(mode="pred").load(val_set),
            epochs=params["refine_epochs"],
        )
        tk.models.save(pred_model, models_dir / "model.h5")
    validate(pred_model)


@app.command()
def validate(pred_model=None):
    train_set, val_set = load_data()
    if pred_model is None:
        _, pred_model = create_network(len(train_set))
        tk.models.load_weights(pred_model, models_dir / "model.h5")
    pred = tk.models.predict(pred_model, MyDataLoader(mode="predict").load(val_set))
    evals = tk.evaluations.evaluate_classification(val_set.labels, pred)
    tk.notifications.post_evals(evals)


def load_data():
    # CSV例:
    # path,noisy_labels_0,noisy_labels_1,noisy_labels_5,noisy_labels_25,noisy_labels_50,is_valid
    # train/n02979186/n02979186_9036.JPEG,n02979186,n02979186,n02979186,n02979186,n02979186,False
    # train/n02979186/n02979186_11957.JPEG,n02979186,n02979186,n02979186,n02979186,n03000684,False
    df = pd.read_csv(data_dir / "noisy_imagenette.csv")
    train_set = _get_data(df[~df["is_valid"]])
    val_set = _get_data(df[df["is_valid"]])
    assert tuple(train_set.metadata["class_names"]) == tuple(
        val_set.metadata["class_names"]
    )
    return train_set, val_set


def _get_data(df):
    data = np.array([data_dir / p for p in df["path"].values])
    labels = df["noisy_labels_50"].values
    class_names = list(sorted(np.unique(labels)))
    assert len(class_names) == params["num_classes"]
    class_name_to_id = {cn: i for i, cn in enumerate(class_names)}
    y = np.array([class_name_to_id[cn] for cn in labels])
    return tk.data.Dataset(data=data, labels=y, metadata={"class_names": class_names})


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
    train_model = tf.keras.models.Model(inputs=inputs, outputs=x)

    global_batch_size = params["batch_size"] * app.num_workers
    schedule = tk.schedules.CosineAnnealing(
        initial_learning_rate=params["base_lr"] * global_batch_size,
        decay_steps=-(-train_size // global_batch_size) * params["epochs"],
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=schedule, momentum=0.9, nesterov=True
    )

    tk.models.compile(
        train_model,
        optimizer,
        loss,
        ["acc", tk.metrics.CosineSimilarity(from_logits=True)],
    )

    x = tf.keras.layers.Activation("softmax")(x)
    pred_model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return train_model, pred_model


def loss(y_true, logits):
    return tk.losses.categorical_crossentropy(
        y_true, logits, from_logits=True, label_smoothing=0.2
    )


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, mode):
        super().__init__(batch_size=params["batch_size"])
        self.mode = mode
        self.aug2: typing.Any = None
        if self.mode == "train":
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        size=params["train_shape"][:2],
                        base_scale=params["predict_shape"][0]
                        / params["train_shape"][0],
                    ),
                    tk.image.RandomColorAugmentors(noisy=False),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
            self.data_per_sample = 2  # mixup
        elif self.mode == "refine":
            self.aug1 = tk.image.RandomTransform.create_refine(
                size=params["predict_shape"][:2]
            )
            self.aug2 = None
        else:
            self.aug1 = tk.image.Resize(size=params["predict_shape"][:2])
            self.aug2 = None

    def get_ds(
        self,
        dataset: tk.data.Dataset,
        shuffle: bool = False,
        without_label: bool = False,
    ) -> tuple[tf.data.Dataset, int]:
        """tf.data.Datasetを作って返す。"""
        X = [str(X_i) for X_i in dataset.data]  # pathlib.Path -> str
        y = dataset.labels

        def process1(X_i, y_i):
            X_i = tf.io.read_file(X_i)
            X_i = tf.io.decode_image(X_i, channels=3, expand_animations=False)
            y_i = tf.one_hot(y_i, params["num_classes"], dtype=tf.float32)
            return X_i, y_i

        def process2(X_i, y_i):
            X_i = tf.numpy_function(
                lambda X_i: self.aug1(image=X_i)["image"], inp=[X_i], Tout=tf.uint8
            )
            X_i = tf.ensure_shape(X_i, params["train_shape"])
            return X_i, y_i

        def process3(X_i, y_i):
            X_i = tf.cast(X_i, tf.float32)
            if self.aug2 is not None:
                X_i = tf.numpy_function(
                    lambda X_i: self.aug2(image=X_i)["image"],
                    inp=[X_i],
                    Tout=tf.float32,
                )
                X_i = tf.ensure_shape(X_i, params["train_shape"])
            X_i = tf.keras.applications.imagenet_utils.preprocess_input(X_i, mode="tf")
            if without_label:
                return X_i
            return X_i, y_i

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if self.mode in ("train", "refine"):
            ds = ds.map(
                process1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle
            )
            assert app.temp_dir is not None
            ds = ds.cache(str(app.temp_dir / f"{self.mode}.cache"))
            ds = ds.shuffle(buffer_size=len(X)) if shuffle else ds
            ds = ds.map(process2)
        else:
            ds = ds.shuffle(buffer_size=len(X)) if shuffle else ds
            ds = ds.map(
                lambda *args: process2(*process1(*args)),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=not shuffle,
            )
        if self.mode == "train":
            assert shuffle
            ds = tk.data.mixup(ds, process3)
        else:
            ds = ds.map(process3)
        ds = ds.repeat() if shuffle else ds  # シャッフル時はバッチサイズを固定するため先にrepeat
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        ds = ds.with_options(options)
        steps = -(-len(dataset) // self.batch_size)
        return ds, steps


if __name__ == "__main__":
    app.run(default="train")
