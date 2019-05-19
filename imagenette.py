#!/usr/bin/env python3
"""imagenetteの実験用コード。"""
import argparse
import pathlib

import numpy as np

import pytoolkit as tk

NUM_CLASSES = 10
INPUT_SHAPE = (321, 321, 3)
BATCH_SIZE = 16

logger = tk.log.get(__name__)


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='train', choices=('check', 'train', 'validate'), nargs='?')
    parser.add_argument('--data-dir', default=pathlib.Path(f'data/imagenette'), type=pathlib.Path)
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
                      batch_size=BATCH_SIZE, epochs=1800, callbacks=callbacks,
                      validation_freq=0,
                      model_path=args.models_dir / 'model.h5')


def validate(args, model=None):
    """検証。"""
    tk.log.init(args.models_dir / f'validate.log')
    _, val_dataset = load_data(args.data_dir)
    model = model or tk.models.load(args.models_dir / 'model.h5')
    pred = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2)
    tk.ml.print_classification_metrics(val_dataset.y, pred)


def load_data(data_dir):
    """データの読み込み。"""
    class_names, X_train, y_train = tk.ml.listup_classification(data_dir / 'train')
    _, X_val, y_val = tk.ml.listup_classification(data_dir / 'val', class_names=class_names)

    # trainとvalを逆にしちゃう。
    (X_train, y_train), (X_val, y_val) = (X_val, y_val), (X_train, y_train)

    train_dataset = MyDataset(X_train, y_train, INPUT_SHAPE, NUM_CLASSES, data_augmentation=True)
    val_dataset = MyDataset(X_val, y_val, INPUT_SHAPE, NUM_CLASSES)
    return train_dataset, val_dataset


def create_model():
    """モデルの作成。"""
    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = tk.layers.Preprocess(mode='tf')(x)
    x = _conv2d(64, 7, strides=2)(x)  # 1/2
    x = _down(128, use_act=False)(x)  # 1/4
    x = _blocks(128, 2)(x)
    x = _down(256, use_act=False)(x)  # 1/8
    x = _blocks(256, 4)(x)
    x = _down(512, use_act=False)(x)  # 1/16
    x = _blocks(512, 8)(x)
    x = _down(512, use_act=False)(x)  # 1/32
    x = _blocks(512, 4)(x)
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
        self.data_augmentation = data_augmentation
        if data_augmentation:
            self.aug1 = tk.image.Compose([
                tk.image.RandomTransform(width=input_shape[1], height=input_shape[0]),
                tk.image.RandomColorAugmentors(),
            ])
            self.aug2 = tk.image.RandomErasing()
        else:
            self.aug1 = tk.image.Resize(width=input_shape[1], height=input_shape[0])
            self.aug2 = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample1 = self.get_sample(index)
        if self.data_augmentation:
            sample2 = self.get_sample(np.random.choice(len(self)))
            X, y = tk.ndimage.mixup(sample1, sample2)
            X = self.aug2(image=X)['image']
        else:
            X, y = sample1
        return X, y

    def get_sample(self, index):
        X = tk.ndimage.load(self.X[index])
        X = self.aug1(image=X)['image']
        y = tk.keras.utils.to_categorical(self.y[index], self.num_classes)
        return X, y


if __name__ == '__main__':
    _main()
