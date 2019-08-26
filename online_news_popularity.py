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
import functools
import pathlib

import numpy as np
import pandas as pd
import sklearn.preprocessing

import pytoolkit as tk

input_shape = (58,)
batch_size = 256
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
    model = create_model(train_dataset.labels.mean())
    tk.training.train(
        model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_preprocessor=MyPreprocessor(data_augmentation=True),
        val_preprocessor=MyPreprocessor(),
        batch_size=batch_size,
        epochs=100,
        callbacks=[tk.callbacks.CosineAnnealing()],
        model_path=models_dir / "model.h5",
        workers=8,
        data_parallel=False,
    )
    pred = tk.models.predict(
        model,
        val_dataset,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
    )
    if tk.hvd.is_master():
        tk.ml.print_regression_metrics(val_dataset.labels, pred)


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def validate(model=None):
    _, val_dataset = load_data()
    model = model or tk.models.load(models_dir / "model.h5")
    pred = tk.models.predict(
        model,
        val_dataset,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
    )
    tk.ml.print_regression_metrics(val_dataset.labels, pred)


def load_data():
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
    X_train = ss.fit_transform(X_train.values)
    X_test = ss.transform(X_test.values)

    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)


def create_model(bias=0):
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

    inputs = x = tk.keras.layers.Input(input_shape)
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
    base_lr = 3e-4 * batch_size * tk.hvd.size()
    optimizer = tk.keras.optimizers.SGD(
        lr=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0
    )
    tk.models.compile(model, optimizer, "mse", ["mae"])
    return model


class MyPreprocessor(tk.data.Preprocessor):
    """Preprocessor。"""

    def __init__(self, data_augmentation=False):
        self.data_augmentation = data_augmentation

    def get_sample(
        self, dataset: tk.data.Dataset, index: int, random: np.random.RandomState
    ):
        sample1 = dataset.get_sample(index)
        if self.data_augmentation:
            sample2 = dataset.get_sample(random.choice(len(dataset)))
            X, y = tk.ndimage.mixup(sample1, sample2, mode="uniform_ex")
        else:
            X, y = sample1
        return X, y


if __name__ == "__main__":
    app.run(default="train")
