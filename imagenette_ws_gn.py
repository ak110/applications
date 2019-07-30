#!/usr/bin/env python3
"""imagenetteの実験用コード。

Weight Standalization + GroupNormalization。重いので普段は使わない。

[INFO ] val_loss: 1.221
[INFO ] val_acc:  0.817

"""
import argparse
import functools
import pathlib

import numpy as np

import pytoolkit as tk

NUM_CLASSES = 10
INPUT_SHAPE = (320, 320, 3)
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
    conv2d = functools.partial(tk.layers.WSConv2D, kernel_size=3)
    bn = functools.partial(tk.layers.GroupNormalization,
                           gamma_regularizer=tk.keras.regularizers.l2(1e-4))
    act = functools.partial(tk.keras.layers.Activation, 'relu')

    def down(filters):
        def layers(x):
            g = tk.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid',
                                       kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
            x = tk.keras.layers.multiply([x, g])
            x = tk.keras.layers.MaxPooling2D(2, strides=1, padding='same')(x)
            x = tk.layers.BlurPooling2D(taps=4)(x)
            x = conv2d(filters)(x)
            x = bn()(x)
            return x
        return layers

    def blocks(filters, count):
        def layers(x):
            for _ in range(count):
                sc = x
                x = conv2d(filters)(x)
                x = bn()(x)
                x = act()(x)
                x = conv2d(filters)(x)
                # resblockのadd前だけgammaの初期値を0にする。 <https://arxiv.org/abs/1812.01187>
                x = bn(gamma_initializer='zeros')(x)
                x = tk.keras.layers.add([sc, x])
            x = bn()(x)
            x = act()(x)
            return x
        return layers

    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = tk.keras.layers.concatenate([  # 1/2
        conv2d(16, kernel_size=2, strides=2)(x),
        conv2d(16, kernel_size=4, strides=2)(x),
        conv2d(16, kernel_size=6, strides=2)(x),
        conv2d(16, kernel_size=8, strides=2)(x),
    ])
    x = bn(groups=16)(x)
    x = act()(x)
    x = conv2d(128, kernel_size=2, strides=2)(x)  # 1/4
    x = bn()(x)
    x = blocks(128, 2)(x)
    x = down(256)(x)  # 1/8
    x = blocks(256, 4)(x)
    x = down(512)(x)  # 1/16
    x = blocks(512, 4)(x)
    x = down(512)(x)  # 1/32
    x = blocks(512, 4)(x)
    x = tk.keras.layers.GlobalAveragePooling2D()(x)
    x = tk.keras.layers.Dense(NUM_CLASSES, activation='softmax',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 1e-3 * BATCH_SIZE * tk.hvd.get().size()
    optimizer = tk.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)
    tk.models.compile(model, optimizer, 'categorical_crossentropy', ['acc'])
    return model


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
        if self.data_augmentation:
            f = tk.threading.get_pool().submit(self.get_sample, np.random.choice(len(self)))
            sample1 = self.get_sample(index)
            sample2 = f.result()
            X, y = tk.ndimage.mixup(sample1, sample2)
            X = self.aug2(image=X)['image']
        else:
            X, y = self.get_sample(index)
        X = tk.ndimage.preprocess_tf(X)
        return X, y

    def get_sample(self, index):
        X = tk.ndimage.load(self.X[index])
        X = self.aug1(image=X)['image']
        y = tk.keras.utils.to_categorical(self.y[index], self.num_classes)
        return X, y


if __name__ == '__main__':
    _main()
