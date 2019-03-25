#!/usr/bin/env python3
"""Train with 1000."""
import argparse
import pathlib

import albumentations as A
import cv2
import numpy as np
import sklearn.metrics

import pytoolkit as tk

logger = tk.log.get(__name__)


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='3epochだけお試し実行(動作確認用)')
    parser.add_argument('--models-dir', default=pathlib.Path('models/train1000'), type=pathlib.Path)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.log.init(args.models_dir / 'train.log')
        _run(args)


@tk.log.trace()
def _run(args):
    epochs = 3 if args.check else 1800
    batch_size = 64
    base_lr = 1e-3 * batch_size * tk.hvd.get().size()

    (X_train, y_train), (X_test, y_test), num_classes = _load_data()
    input_shape = X_train.shape[1:]
    train_dataset = MyDataset(X_train, y_train, input_shape, num_classes, data_augmentation=True)
    test_dataset = MyDataset(X_test, y_test, input_shape, num_classes)
    train_data = tk.data.DataLoader(train_dataset, batch_size, shuffle=True, mixup=True, mp_size=tk.hvd.get().size())
    test_data = tk.data.DataLoader(test_dataset, batch_size * 2)

    model = _create_network(input_shape, num_classes)
    optimizer = tk.optimizers.NSGD(lr=base_lr, momentum=0.9, nesterov=True)
    optimizer = tk.hvd.get().DistributedOptimizer(optimizer, compression=tk.hvd.get().Compression.fp16)
    model.compile(optimizer, 'categorical_crossentropy', ['acc'])
    model.summary(print_fn=logger.info if tk.hvd.is_master() else lambda x: x)

    callbacks = [
        tk.callbacks.CosineAnnealing(),
        tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0),
        tk.hvd.get().callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    ]
    model.fit_generator(train_data, epochs=epochs, callbacks=callbacks,
                        verbose=1 if tk.hvd.is_master() else 0)

    if tk.hvd.is_master():
        # 検証
        pred_test = model.predict_generator(test_data, verbose=1 if tk.hvd.is_master() else 0)
        logger.info(f'Test Accuracy:      {sklearn.metrics.accuracy_score(y_test, pred_test.argmax(axis=-1)):.4f}')
        logger.info(f'Test Cross Entropy: {sklearn.metrics.log_loss(y_test, pred_test):.4f}')
        # 後で何かしたくなった時のために一応保存
        model.save(args.models_dir / 'model.h5', include_optimizer=False)


def _load_data():
    """データの読み込み。"""
    (X_train, y_train), (X_test, y_test) = tk.keras.datasets.cifar10.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    num_classes = len(np.unique(y_train))
    X_train, y_train = _extract1000(X_train, y_train, num_classes=num_classes)
    return (X_train, y_train), (X_test, y_test), num_classes


def _extract1000(X, y, num_classes):
    """https://github.com/mastnk/train1000 を参考にクラスごとに均等に先頭から取得する処理。"""
    num_data = 1000
    num_per_class = num_data // num_classes

    index_list = []
    for c in range(num_classes):
        index_list.extend(np.where(y == c)[0][:num_per_class])
    assert len(index_list) == num_data

    return X[index_list], y[index_list]


def _create_network(input_shape, num_classes):
    """ネットワークを作成して返す。"""
    inputs = x = tk.keras.layers.Input(input_shape)
    x = tk.layers.Preprocess(mode='tf')(x)
    x = _conv2d(128, use_act=False)(x)
    x = _blocks(128, 8)(x)
    x = _conv2d(256, strides=2, use_act=False)(x)
    x = _blocks(256, 8)(x)
    x = _conv2d(512, strides=2, use_act=False)(x)
    x = _blocks(512, 8)(x)
    x = tk.keras.layers.GlobalAveragePooling2D()(x)
    x = tk.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    return model


def _blocks(filters, count):
    def layers(x):
        for _ in range(count):
            sc = x
            x = _conv2d(filters, use_act=True)(x)
            x = _conv2d(filters, use_act=False)(x)
            x = tk.keras.layers.add([sc, x])
        x = _bn_act()(x)
        return x
    return layers


def _conv2d(filters, kernel_size=3, strides=1, use_act=True):
    def layers(x):
        x = tk.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides,
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_uniform',
                                   kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
        x = _bn_act(use_act=use_act)(x)
        return x
    return layers


def _bn_act(use_act=True):
    def layers(x):
        x = tk.keras.layers.BatchNormalization(gamma_regularizer=tk.keras.regularizers.l2(1e-4))(x)
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
            self.aug = A.Compose([
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
