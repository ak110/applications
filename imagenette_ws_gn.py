#!/usr/bin/env python3
"""imagenetteの実験用コード。"""
import argparse
import pathlib

import pytoolkit as tk

logger = tk.log.get(__name__)


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='train', choices=('check', 'train'), nargs='?')
    parser.add_argument('--data-dir', default=pathlib.Path(f'data/imagenette'), type=pathlib.Path)
    parser.add_argument('--models-dir', default=pathlib.Path(f'models/{pathlib.Path(__file__).stem}'), type=pathlib.Path)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=args.mode not in ('check',)):
        tk.log.init(None if args.mode in ('check',) else args.models_dir / f'{args.mode}.log')
        {
            'check': _check,
            'train': _train,
        }[args.mode](args)


@tk.log.trace()
def _check(args):
    _ = args
    num_classes = 10
    input_shape = (321, 321, 3)
    model = _create_network(input_shape, num_classes)
    model.summary()


@tk.log.trace()
def _train(args):
    input_shape = (321, 321, 3)
    epochs = 1800
    batch_size = 16
    base_lr = 1e-3 * batch_size * tk.hvd.get().size()

    (X_train, y_train), (X_val, y_val), class_names = _load_data(args.data_dir)
    num_classes = len(class_names)
    train_dataset = MyDataset(X_train, y_train, input_shape, num_classes, data_augmentation=True)
    val_dataset = MyDataset(X_val, y_val, input_shape, num_classes)

    model = _create_network(input_shape, num_classes)
    optimizer = tk.optimizers.NSGD(lr=base_lr, momentum=0.9, nesterov=True)
    tk.models.compile(model, optimizer, 'categorical_crossentropy', ['acc'])
    tk.models.summary(model)
    tk.models.plot_model(model, args.models_dir / 'model.svg', show_shapes=True)

    callbacks = [
        tk.callbacks.CosineAnnealing(),
        tk.hvd.get().callbacks.BroadcastGlobalVariablesCallback(0),
        tk.hvd.get().callbacks.MetricAverageCallback(),
        tk.hvd.get().callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    ]
    tk.models.fit(model, train_dataset, batch_size=batch_size,
                  epochs=epochs, verbose=1, callbacks=callbacks,
                  mixup=True)
    # 後で何かしたくなった時のために一応保存
    tk.models.save(model, args.models_dir / 'model.h5')

    evals = tk.models.evaluate(model, val_dataset, batch_size=batch_size * 2)
    if tk.hvd.is_master():
        for n, v in evals.items():
            logger.info(f'{n:8s}: {v:.3f}')


def _load_data(data_dir):
    """データの読み込み。"""
    class_names, X_train, y_train = tk.ml.listup_classification(data_dir / 'train')
    _, X_val, y_val = tk.ml.listup_classification(data_dir / 'val', class_names=class_names)

    # trainとvalを逆にしちゃう。
    (X_train, y_train), (X_val, y_val) = (X_val, y_val), (X_train, y_train)

    return (X_train, y_train), (X_val, y_val), class_names


def _create_network(input_shape, num_classes):
    """ネットワークを作成して返す。"""
    inputs = x = tk.keras.layers.Input(input_shape)
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
    x = tk.keras.layers.Dense(num_classes, activation='softmax',
                              kernel_regularizer=tk.keras.regularizers.l2(1e-4))(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
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
        x = tk.layers.WSConv2D(filters, kernel_size=kernel_size, strides=strides)(x)
        x = _bn_act(use_act=use_act, gamma_zero=gamma_zero)(x)
        return x
    return layers


def _bn_act(use_act=True, gamma_zero=False):
    def layers(x):
        # resblockのadd前だけgammaの初期値を0にする。 <https://arxiv.org/abs/1812.01187>
        x = tk.layers.GroupNormalization(gamma_initializer='zeros' if gamma_zero else 'ones',
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
                tk.image.RandomTransform(width=input_shape[1], height=input_shape[0]),
                tk.image.RandomColorAugmentors(),
                tk.image.RandomErasing(),
            ])
        else:
            self.aug = tk.image.Resize(width=input_shape[1], height=input_shape[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = tk.ndimage.load(self.X[index])
        X = self.aug(image=X)['image']
        y = tk.keras.utils.to_categorical(self.y[index], self.num_classes)
        return X, y


if __name__ == '__main__':
    _main()
