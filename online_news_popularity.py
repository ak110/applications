#!/usr/bin/env python3
"""某online_news_popularityの実験用コード。

参考:

<https://github.com/ozt-ca/tjo.hatenablog.samples/tree/master/r_samples/public_lib/jp/exp_uci_datasets/online_news_popularity>

<https://gist.github.com/KazukiOnodera/64ffa671d47df059f97051b58e8bc32c>
0.8609987920839894

実行結果:
```
[INFO ] R^2:  0.142
[INFO ] RMSE: 0.875 (base: 0.945)
[INFO ] MAE:  0.637 (base: 0.709)
```

"""
import argparse
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
    parser.add_argument('mode', default='train', choices=('check', 'train', 'validate'), nargs='?')
    parser.add_argument('--data-dir', default=pathlib.Path(f'data/online_news_popularity'), type=pathlib.Path)
    parser.add_argument('--models-dir', default=pathlib.Path(f'models/{pathlib.Path(__file__).stem}'), type=pathlib.Path)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.utils.find_by_name([check, train, validate], args.mode)(args)


def check(args):
    """動作確認用コード。"""
    tk.log.init(None)
    model = create_model()
    tk.training.check(model, plot_path=args.models_dir / 'model.svg')


def train(args):
    """学習。"""
    tk.log.init(args.models_dir / f'train.log')
    train_dataset, val_dataset = load_data(args.data_dir)
    model = create_model()
    callbacks = []
    callbacks.append(tk.callbacks.CosineAnnealing())
    tk.training.train(model, train_dataset, val_dataset,
                      batch_size=BATCH_SIZE, epochs=100, callbacks=callbacks,
                      model_path=args.models_dir / 'model.h5',
                      workers=8, data_parallel=False)
    pred = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2, use_horovod=True)
    pred = train_dataset.ss2.inverse_transform(pred)
    if tk.hvd.is_master():
        tk.ml.print_regression_metrics(val_dataset.y, pred)


def validate(args, model=None):
    """検証。"""
    tk.log.init(args.models_dir / f'validate.log')
    train_dataset, val_dataset = load_data(args.data_dir)
    model = model or tk.models.load(args.models_dir / 'model.h5')
    pred = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2)
    pred = train_dataset.ss2.inverse_transform(pred)
    tk.ml.print_regression_metrics(val_dataset.y, pred)


def load_data(data_dir):
    """データの読み込み。"""
    df_train = pd.read_csv(data_dir / 'ONP_train.csv')
    df_test = pd.read_csv(data_dir / 'ONP_test.csv')
    y_train = df_train['shares']
    X_train = df_train.drop('shares', axis=1)
    y_test = df_test['shares']
    X_test = df_test.drop('shares', axis=1)

    ss1 = sklearn.preprocessing.StandardScaler()
    ss2 = sklearn.preprocessing.StandardScaler()
    ss1.fit(X_train)
    ss2.fit(y_train.values.reshape(-1, 1))

    train_dataset = MyDataset(X_train.values, y_train.values, ss1, ss2, data_augmentation=True)
    test_dataset = MyDataset(X_test.values, y_test.values, ss1, ss2)
    return train_dataset, test_dataset


def create_model():
    """モデルの作成。"""
    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = tk.keras.layers.Dense(512,
                              kernel_initializer='he_uniform',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    x = tk.layers.MixFeat()(x)
    x = tk.keras.layers.Activation('elu')(x)
    x = tk.keras.layers.Dense(512,
                              kernel_initializer='he_uniform',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    x = tk.layers.MixFeat()(x)
    x = tk.keras.layers.Activation('elu')(x)
    x = tk.keras.layers.Dense(1, kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 1e-5 * BATCH_SIZE * tk.hvd.get().size()
    optimizer = tk.optimizers.NSGD(lr=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0)
    tk.models.compile(model, optimizer, 'mse', ['mae'])
    return model


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, ss1, ss2, data_augmentation=False):
        self.X = X
        self.y = y
        self.X_tr = ss1.transform(X)
        self.y_tr = ss2.transform(y.reshape(-1, 1))[:, 0]
        self.ss1 = ss1
        self.ss2 = ss2
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample1 = self.get_sample(index)
        if self.data_augmentation:
            sample2 = self.get_sample(np.random.choice(len(self)))
            X, y = tk.ndimage.mixup(sample1, sample2, mode='uniform_ex')
        else:
            X, y = sample1
        return X, y

    def get_sample(self, index):
        return self.X_tr[index], self.y_tr[index]


if __name__ == '__main__':
    _main()
