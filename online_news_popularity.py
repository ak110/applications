#!/usr/bin/env python3
"""某online_news_popularityの実験用コード。

参考:

<https://github.com/ozt-ca/tjo.hatenablog.samples/tree/master/r_samples/public_lib/jp/exp_uci_sets/online_news_popularity>

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

import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

import pytoolkit as tk

input_shape = (58,)
batch_size = 256
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
    evals = model.train(train_set, val_set)
    pred = model.predict(val_set, fold=0)
    if tk.hvd.is_master():
        evals = tk.evaluations.print_regression(val_set.labels, pred)
        tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate(model=None):
    _, val_set = load_data()
    model = create_model().load()
    pred = model.predict(val_set, fold=0)
    if tk.hvd.is_master():
        tk.evaluations.print_regression(val_set.labels, pred)


def load_data():
    df_train = pd.read_csv(
        "https://github.com/ozt-ca/tjo.hatenablog.samples"
        "/raw/master/r_samples/public_lib/jp/exp_uci_sets/online_news_popularity/ONP_train.csv"
    )
    df_test = pd.read_csv(
        "https://github.com/ozt-ca/tjo.hatenablog.samples"
        "/raw/master/r_samples/public_lib/jp/exp_uci_sets/online_news_popularity/ONP_test.csv"
    )
    y_train = df_train["shares"]
    X_train = df_train.drop("shares", axis=1)
    y_test = df_test["shares"]
    X_test = df_test.drop("shares", axis=1)

    ss = sklearn.preprocessing.StandardScaler()
    X_train = ss.fit_transform(X_train.values)
    X_test = ss.transform(X_test.values)

    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        nfold=1,
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        epochs=100,
        callbacks=[tk.callbacks.CosineAnnealing()],
        fit_params={"workers": 8},
        models_dir=models_dir,
        model_name_format="model.h5",
        skip_if_exists=False,
    )


def create_network():
    dense = functools.partial(
        tf.keras.layers.Dense,
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tf.keras.layers.BatchNormalization,
        gamma_regularizer=tf.keras.regularizers.l2(1e-5),
    )
    act = functools.partial(tf.keras.layers.Activation, activation="elu")

    inputs = x = tf.keras.layers.Input(input_shape)
    x = dense(512)(x)
    for _ in range(3):
        sc = x
        x = bn()(x)
        x = act()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = dense(512)(x)
        x = bn()(x)
        x = act()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = dense(512, kernel_initializer="zeros")(x)
        x = tf.keras.layers.add([sc, x])
    x = bn()(x)
    x = act()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    base_lr = 3e-4 * batch_size * tk.hvd.size()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0
    )

    tk.models.compile(model, optimizer, "mse", ["mae"])
    return model, model


class MyDataLoader(tk.data.DataLoader):
    def __init__(self, data_augmentation=False):
        super().__init__(
            batch_size=batch_size, data_per_sample=2 if data_augmentation else 1
        )
        self.data_augmentation = data_augmentation

    def get_data(self, dataset: tk.data.Dataset, index: int):
        return dataset.get_data(index)

    def get_sample(self, data):
        if self.data_augmentation:
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
        else:
            X, y = super().get_sample(data)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
