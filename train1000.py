#!/usr/bin/env python3
"""Train with 1000."""
import argparse
import pathlib

import albumentations as A
import cv2
import numpy as np

import pytoolkit as tk

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 64

logger = tk.log.get(__name__)


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='train', choices=('check', 'train', 'validate'), nargs='?')
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
    train_dataset, val_dataset = load_data()
    model = create_model()
    callbacks = []
    callbacks.append(tk.callbacks.CosineAnnealing())
    tk.training.train(model, train_dataset, val_dataset,
                      batch_size=BATCH_SIZE, epochs=3, callbacks=callbacks,
                      mixup=True, validation_freq=0,
                      model_path=args.models_dir / 'model.h5')


def validate(args, model=None):
    """検証。"""
    tk.log.init(args.models_dir / f'validate.log')
    _, val_dataset = load_data()
    model = model or tk.models.load(args.models_dir / 'model.h5')
    pred = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2)
    tk.ml.print_classification_metrics(val_dataset.y, pred)


def load_data():
    """データの読み込み。"""
    (X_train, y_train), (X_val, y_val) = tk.keras.datasets.cifar10.load_data()
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)
    num_classes = len(np.unique(y_train))
    X_train, y_train = tk.ml.extract1000(X_train, y_train, num_classes=num_classes)
    train_dataset = MyDataset(X_train, y_train, INPUT_SHAPE, NUM_CLASSES, data_augmentation=True)
    val_dataset = MyDataset(X_val, y_val, INPUT_SHAPE, NUM_CLASSES)
    return train_dataset, val_dataset


def create_model():
    """モデルの作成。"""
    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = tk.layers.Preprocess(mode='tf')(x)
    x = _conv2d(128, use_act=False)(x)
    x = _blocks(128, 8)(x)
    x = _down(256, use_act=False)(x)
    x = _blocks(256, 8)(x)
    x = _down(512, use_act=False)(x)
    x = _blocks(512, 8)(x)
    x = tk.keras.layers.GlobalAveragePooling2D()(x)
    x = tk.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 1e-3 * BATCH_SIZE * tk.hvd.get().size()
    optimizer = tk.optimizers.NSGD(lr=base_lr, momentum=0.9, nesterov=True)
    tk.models.compile(model, optimizer, 'categorical_crossentropy', ['acc'])
    return model


def _down(filters, use_act=True):
    def layers(x):
        g = tk.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
        x = tk.keras.layers.multiply([x, g])
        x = _conv2d(filters, strides=2, use_act=use_act)(x)
        return x
    return layers


def _blocks(filters, count):
    def layers(x):
        for _ in range(count):
            sc = x
            x = _conv2d(filters)(x)
            x = _conv2d(filters, use_act=False, gamma_zero=True)(x)
            x = tk.keras.layers.add([sc, x])
        x = _bn_act()(x)
        return x
    return layers


def _conv2d(filters, kernel_size=3, strides=1, use_act=True, gamma_zero=False):
    def layers(x):
        x = tk.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
        x = _bn_act(use_act=use_act, gamma_zero=gamma_zero)(x)
        return x
    return layers


def _bn_act(use_act=True, gamma_zero=False):
    def layers(x):
        # resblockのadd前だけgammaの初期値を0にする。 <https://arxiv.org/abs/1812.01187>
        x = tk.keras.layers.BatchNormalization(gamma_initializer='zeros' if gamma_zero else 'ones',
                                               gamma_regularizer=tk.keras.regularizers.l2(1e-4))(x)
        x = tk.layers.MixFeat()(x)
        x = tk.keras.layers.Activation('relu')(x) if use_act else x
        return x
    return layers


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, input_shape, num_classes, data_augmentation=False):
        self.X = X
        self.y = y
        self.input_shape = input_shape
        self.num_classes = num_classes
        if data_augmentation:
            self.aug = tk.image.Compose([
                A.PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, value=[127, 127, 127], p=1),
                tk.autoaugment.CIFAR10Policy(),
                A.RandomCrop(32, 32),
                tk.image.RandomErasing(),
            ])
        else:
            self.aug = A.Compose([])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.aug(image=self.X[index])['image']
        y = tk.keras.utils.to_categorical(self.y[index], self.num_classes)
        return X, y


if __name__ == '__main__':
    _main()
