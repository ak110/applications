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
import argparse
import pathlib

import numpy as np
import pandas as pd

import pytoolkit as tk

INPUT_SHAPE = (512,)
BATCH_SIZE = 32
NUM_CLASSES = 8

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
    model = create_model()
    callbacks = []
    callbacks.append(tk.callbacks.CosineAnnealing())
    tk.training.train(
        model,
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE,
        epochs=20,
        callbacks=callbacks,
        model_path=args.models_dir / "model.h5",
        workers=8,
        data_parallel=False,
    )
    pred = tk.models.predict(
        model, val_dataset, batch_size=BATCH_SIZE * 2, use_horovod=True
    )
    if tk.hvd.is_master():
        tk.ml.print_classification_metrics(val_dataset.y, pred)


def validate(args, model=None):
    """検証。"""
    tk.log.init(args.models_dir / f"validate.log")
    _, val_dataset = load_data()
    model = model or tk.models.load(args.models_dir / "model.h5")
    pred = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2)
    tk.ml.print_classification_metrics(val_dataset.y, pred)


def load_data():
    """データの読み込み。"""
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
    assert len(class_names) == NUM_CLASSES
    class_to_id = np.vectorize({c: i for i, c in enumerate(class_names)}.__getitem__)

    X_train = df_train["text"].values
    y_train = class_to_id(df_train["class"].values)
    X_test = df_test["text"].values
    y_test = class_to_id(df_test["class"].values)

    train_dataset = MyDataset(
        X_train, y_train, NUM_CLASSES, INPUT_SHAPE, data_augmentation=True
    )
    test_dataset = MyDataset(X_test, y_test, NUM_CLASSES, INPUT_SHAPE)
    return train_dataset, test_dataset


def create_model():
    """モデルの作成。"""
    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = tk.keras.layers.Embedding(65536, 256, mask_zero=True)(x)
    x1 = tk.keras.layers.GlobalAveragePooling1D()(x)
    x2 = tk.keras.layers.GlobalMaxPooling1D()(tk.layers.RemoveMask()(x))
    x = tk.keras.layers.concatenate([x1, x2])
    x = tk.keras.layers.Dense(
        NUM_CLASSES,
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
        activation="softmax",
    )(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return model


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, num_classes, input_shape, data_augmentation=False):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = np.frombuffer(
            self.X[index].replace(" ", "").encode("utf-16-le"), dtype=np.uint16
        )
        X = tk.keras.preprocessing.sequence.pad_sequences([X], self.input_shape[0])[0]
        y = tk.keras.utils.to_categorical(self.y[index], self.num_classes)
        return X, y


if __name__ == "__main__":
    _main()
