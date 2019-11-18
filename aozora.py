#!/usr/bin/env python3
"""某青空文庫の実験用コード。

参考:

<https://github.com/ozt-ca/tjo.hatenablog.samples/tree/master/r_samples/public_lib/jp/aozora>

<https://tjo.hatenablog.com/entry/2019/05/31/190000>
AP:   0.945
Prec: 0.88742
Rec:  0.86312

実行結果:
```
[INFO ] Accuracy:   0.892 (Error: 0.108)
[INFO ] F1-macro:   0.894
[INFO ] AUC-macro:  0.990
[INFO ] AP-macro:   0.957
[INFO ] Prec-macro: 0.898
[INFO ] Rec-macro:  0.892
[INFO ] Logloss:    0.339
```

"""
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

import pytoolkit as tk

input_shape = (512,)
batch_size = 32
num_classes = 8
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
    pred = model.predict(val_set)[0]
    if tk.hvd.is_master():
        evals = tk.evaluations.print_classification_metrics(val_set.labels, pred)
        tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate():
    _, val_set = load_data()
    model = create_model().load(models_dir)
    pred = model.predict(val_set)[0]
    tk.evaluations.print_classification_metrics(val_set.labels, pred)


def load_data():
    df_train = pd.read_csv(
        "https://raw.githubusercontent.com/ozt-ca/tjo.hatenablog.samples/master/r_samples/public_lib/jp/aozora/aozora_8writers_train.csv",
        header=None,
        names=["text", "class"],
    )
    df_test = pd.read_csv(
        "https://raw.githubusercontent.com/ozt-ca/tjo.hatenablog.samples/master/r_samples/public_lib/jp/aozora/aozora_8writers_test.csv",
        header=None,
        names=["text", "class"],
    )

    class_names = list(sorted(np.unique(df_train["class"].values)))
    assert len(class_names) == num_classes
    class_to_id = np.vectorize({c: i for i, c in enumerate(class_names)}.__getitem__)

    X_train = df_train["text"].values
    y_train = class_to_id(df_train["class"].values)
    X_test = df_test["text"].values
    y_test = class_to_id(df_test["class"].values)

    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_test, y_test)


def create_model():
    return tk.pipeline.KerasModel(
        create_network_fn=create_network,
        nfold=1,
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        epochs=20,
        callbacks=[tk.callbacks.CosineAnnealing()],
        fit_params={"workers": 8},
        models_dir=models_dir,
        model_name_format="model.h5",
        skip_if_exists=False,
        use_horovod=True,
    )


def create_network() -> tf.keras.models.Model:
    inputs = x = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Embedding(65536, 256, mask_zero=True)(x)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x)
    x2 = tf.keras.layers.GlobalMaxPooling1D()(tk.layers.RemoveMask()(x))
    x = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        activation="softmax",
    )(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    tk.models.compile(model, "adam", "categorical_crossentropy", ["acc"])
    return model


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, data_augmentation=False):
        super().__init__(batch_size=batch_size, parallel=False)
        self.data_augmentation = data_augmentation

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = np.frombuffer(X.replace(" ", "").encode("utf-16-le"), dtype=np.uint16)
        X = tf.keras.preprocessing.sequence.pad_sequences([X], input_shape[0])[0]
        y = tf.keras.utils.to_categorical(y, num_classes)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
