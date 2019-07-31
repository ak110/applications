#!/usr/bin/env python3
"""某online_news_popularityの実験用コード。

参考:

<https://github.com/ozt-ca/tjo.hatenablog.samples/tree/master/r_samples/public_lib/jp/exp_uci_datasets/online_news_popularity>

<https://gist.github.com/KazukiOnodera/64ffa671d47df059f97051b58e8bc32c>
0.8609987920839894

実行結果:
```
[INFO ] R^2:      0.154
[INFO ] RMSE:     0.869 (base: 0.945)
[INFO ] MAE:      0.635 (base: 0.710)
[INFO ] RMSE/MAE: 1.368
```

"""
import argparse
import functools
import pathlib

import numpy as np
import pandas as pd
import sklearn.preprocessing

import pytoolkit as tk

INPUT_SHAPE = (58,)
BATCH_SIZE = 256

logger = tk.log.get(__name__)


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", default="train", choices=("check", "train", "validate"), nargs="?"
    )
    parser.add_argument(
        "--models-dir",
        default=pathlib.Path(f"models/{pathlib.Path(__file__).stem}"),
        type=pathlib.Path,
    )
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.utils.find_by_name([check, train, validate], args.mode)(args)


def check(args):
    """動作確認用コード。"""
    tk.log.init(None)
    model = create_model()
    tk.training.check(model, plot_path=args.models_dir / "model.svg")


def train(args):
    """学習。"""
    tk.log.init(args.models_dir / f"train.log")
    train_dataset, val_dataset = load_data()
    model = create_model(train_dataset.y.mean())
    callbacks = []
    callbacks.append(tk.callbacks.CosineAnnealing())
    tk.training.train(
        model,
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE,
        epochs=100,
        callbacks=callbacks,
        model_path=args.models_dir / "model.h5",
        workers=8,
        data_parallel=False,
    )
    pred = tk.models.predict(
        model, val_dataset, batch_size=BATCH_SIZE * 2, use_horovod=True
    )
    if tk.hvd.is_master():
        tk.ml.print_regression_metrics(val_dataset.y, pred)


def validate(args, model=None):
    """検証。"""
    tk.log.init(args.models_dir / f"validate.log")
    _, val_dataset = load_data()
    model = model or tk.models.load(args.models_dir / "model.h5")
    pred = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2)
    tk.ml.print_regression_metrics(val_dataset.y, pred)


def load_data():
    """データの読み込み。"""
    df_train = pd.read_csv(
        "https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/exp_uci_datasets/online_news_popularity/ONP_train.csv"
    )
    df_test = pd.read_csv(
        "https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/exp_uci_datasets/online_news_popularity/ONP_test.csv"
    )
    y_train = df_train["shares"]
    X_train = df_train.drop("shares", axis=1)
    y_test = df_test["shares"]
    X_test = df_test.drop("shares", axis=1)

    ss = sklearn.preprocessing.StandardScaler()
    ss.fit(X_train)

    train_dataset = MyDataset(
        X_train.values, y_train.values, ss, data_augmentation=True
    )
    test_dataset = MyDataset(X_test.values, y_test.values, ss)
    return train_dataset, test_dataset


def create_model(bias=0):
    """モデルの作成。"""
    dense = functools.partial(
        tk.keras.layers.Dense,
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tk.keras.layers.BatchNormalization,
        gamma_regularizer=tk.keras.regularizers.l2(1e-5),
    )
    act = functools.partial(tk.keras.layers.Activation, activation="elu")

    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = dense(512)(x)
    for _ in range(3):
        sc = x
        x = bn()(x)
        x = act()(x)
        x = tk.keras.layers.Dropout(0.5)(x)
        x = dense(512)(x)
        x = bn()(x)
        x = act()(x)
        x = tk.keras.layers.Dropout(0.5)(x)
        x = dense(512, kernel_initializer="zeros")(x)
        x = tk.keras.layers.add([sc, x])
    x = bn()(x)
    x = act()(x)
    x = tk.keras.layers.Dropout(0.5)(x)
    x = tk.keras.layers.Dense(
        1,
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
        bias_initializer=tk.keras.initializers.constant(bias),
    )(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 3e-4 * BATCH_SIZE * tk.hvd.get().size()
    optimizer = tk.keras.optimizers.SGD(
        lr=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0
    )
    tk.models.compile(model, optimizer, "mse", ["mae"])
    return model


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, ss, data_augmentation=False):
        self.X = X
        self.y = y
        self.X_tr = ss.transform(X)
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample1 = self.get_sample(index)
        if self.data_augmentation:
            sample2 = self.get_sample(np.random.choice(len(self)))
            X, y = tk.ndimage.mixup(sample1, sample2, mode="uniform_ex")
        else:
            X, y = sample1
        return X, y

    def get_sample(self, index):
        return self.X_tr[index], self.y[index]


if __name__ == "__main__":
    _main()
