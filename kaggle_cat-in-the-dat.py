#!/usr/bin/env python3
"""Categorical Feature Encoding Challengeの実験用コード。"""
import pathlib

import numpy as np
import pandas as pd
import sklearn.metrics

import pytoolkit as tk

nfold = 5
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,
    "nthread": -1,
    # "verbosity": -1,
    # "max_bin": 511,
    # "num_leaves": 31,
    # "min_data_in_leaf": 10,
    "feature_fraction": "sqrt",
    "bagging_freq": 0,
    # "max_depth": 4,
}
seeds = np.arange(5) + 1
# split_seeds = np.arange(5) + 1
data_dir = pathlib.Path(f"data/kaggle_cat-in-the-dat")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command()
def train():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, split_seed=1)
    create_pipeline().cv(train_set, folds, models_dir)
    validate()


@app.command()
def validate():
    train_set = load_train_data()
    folds = tk.validation.split(train_set, nfold, split_seed=1)
    oofp = create_pipeline().load(models_dir).predict_oof(train_set, folds)
    tk.notification.post({"auc": sklearn.metrics.roc_auc_score(train_set.labels, oofp)})
    predict()


@app.command()
def predict():
    # TODO: ValueError: Unknown values in column 'nom_8': {'2be51c868', '1f0a80e1d', 'ec337ce4c', 'a9bf3dc47'}
    test_set = load_test_data()
    pred = create_pipeline().load(models_dir).predict(test_set)
    df = pd.DataFrame()
    df["id"] = test_set.ids
    df["target"] = pred
    df.to_csv(models_dir / "submission.csv", index=False)


def load_train_data():
    df = pd.read_csv(data_dir / "train.csv")
    return _preprocess(df)


def load_test_data():
    df = pd.read_csv(data_dir / "test.csv")
    return _preprocess(df)


def _preprocess(df):
    df["bin_0"] = tk.preprocessing.encode_binary(df["bin_0"], 1, 0)
    df["bin_1"] = tk.preprocessing.encode_binary(df["bin_1"], 1, 0)
    df["bin_2"] = tk.preprocessing.encode_binary(df["bin_2"], 1, 0)
    df["bin_3"] = tk.preprocessing.encode_binary(df["bin_3"], "T", "F")
    df["bin_4"] = tk.preprocessing.encode_binary(df["bin_4"], "Y", "N")
    df["ord_1"] = tk.preprocessing.encode_ordinal(
        df["ord_1"], ["Novice", "Contributor", "Expert", "Master", "Grandmaster"]
    )
    df["ord_2"] = tk.preprocessing.encode_ordinal(
        df["ord_2"], ["Freezing", "Cold", "Warm", "Hot", "Boiling Hot", "Lava Hot"]
    )
    df["ord_3"] = df["ord_3"].map(ord).astype(np.int32)
    df["ord_4"] = df["ord_4"].map(ord).astype(np.int32)
    df["ord_5"] = (
        df["ord_5"].apply(lambda s: ord(s[0]) * 255 + ord(s[1])).astype(np.int32)
    )
    df[["day_sin", "day_cos"]] = tk.preprocessing.encode_cyclic(df["day"], 1, 7 + 1)
    df[["month_sin", "month_cos"]] = tk.preprocessing.encode_cyclic(
        df["month"], 1, 12 + 1
    )
    if "target" in df.columns.values:
        return tk.data.Dataset(
            data=df.drop(columns=["target"]), labels=df["target"].values
        )
    else:
        return tk.data.Dataset(data=df)


def create_pipeline():
    return tk.pipeline.LGBModel(
        params, nfold, seeds=seeds, preprocessors=[tk.preprocessing.FeaturesEncoder()]
    )


if __name__ == "__main__":
    app.run(default="train")
