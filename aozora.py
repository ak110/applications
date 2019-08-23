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

import pytoolkit as tk

input_shape = (512,)
batch_size = 32
num_classes = 8
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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_preprocessor=MyPreprocessor(data_augmentation=True),
        val_preprocessor=MyPreprocessor(),
        batch_size=batch_size,
        epochs=20,
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
        tk.ml.print_classification_metrics(val_dataset.labels, pred)


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
    tk.ml.print_classification_metrics(val_dataset.labels, pred)


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
    inputs = x = tk.keras.layers.Input(input_shape)
    x = tk.keras.layers.Embedding(65536, 256, mask_zero=True)(x)
    x1 = tk.keras.layers.GlobalAveragePooling1D()(x)
    x2 = tk.keras.layers.GlobalMaxPooling1D()(tk.layers.RemoveMask()(x))
    x = tk.keras.layers.concatenate([x1, x2])
    x = tk.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
        activation="softmax",
    )(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


class MyPreprocessor(tk.data.Preprocessor):
    """Preprocessor。"""

    def __init__(self, data_augmentation=False):
        self.data_augmentation = data_augmentation

    def get_sample(
        self, dataset: tk.data.Dataset, index: int, random: np.random.RandomState
    ):
        X, y = dataset.get_sample(index)
        X = np.frombuffer(X.replace(" ", "").encode("utf-16-le"), dtype=np.uint16)
        X = tk.keras.preprocessing.sequence.pad_sequences([X], input_shape[0])[0]
        y = tk.keras.utils.to_categorical(y, num_classes)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
