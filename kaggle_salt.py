#!/usr/bin/env python3
"""TGS Salt Identification Challengeの実験用コード。

- 一番良かったモデル(ls_darknet53_coord_hcs): 0.871
- 転移学習無しの最大(ls_scratch): 0.837

"""
import argparse
import pathlib

import numpy as np

import pytoolkit as tk

INPUT_SHAPE = (101, 101, 1)
BATCH_SIZE = 16

logger = tk.log.get(__name__)


def _main():
    tk.utils.better_exceptions()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='train', choices=('check', 'train', 'validate'), nargs='?')
    parser.add_argument('--data-dir', default=pathlib.Path(f'data/kaggle_salt'), type=pathlib.Path)
    parser.add_argument('--models-dir', default=pathlib.Path(f'models/{pathlib.Path(__file__).stem}'), type=pathlib.Path)
    args = parser.parse_args()
    with tk.dl.session(use_horovod=True):
        tk.utils.find_by_name([check, train, validate], args.mode)(args)


@tk.log.trace()
def check(args):
    """動作確認用コード。"""
    tk.log.init(None)
    model = create_model()
    tk.models.summary(model)
    tk.models.plot(model, args.models_dir / 'model.svg')


@tk.log.trace()
def train(args):
    """学習。"""
    tk.log.init(args.models_dir / f'train.log')
    train_dataset, val_dataset = load_data(args.data_dir)
    model = create_model()
    callbacks = []
    callbacks.append(tk.callbacks.CosineAnnealing())
    tk.models.fit(model, train_dataset, validation_data=val_dataset, batch_size=BATCH_SIZE,
                  epochs=600, verbose=1, callbacks=callbacks)
    tk.models.save(model, args.models_dir / 'model.h5')
    tk.models.evaluate(model, val_dataset, batch_size=BATCH_SIZE * 2)
    _evaluate(model, val_dataset)


@tk.log.trace()
def validate(args, model=None):
    """検証。"""
    tk.log.init(args.models_dir / f'validate.log')
    _, val_dataset = load_data(args.data_dir)
    model = model or tk.models.load(args.models_dir / 'model.h5')
    _evaluate(model, val_dataset)


def _evaluate(model, val_dataset):
    pred_val = tk.models.predict(model, val_dataset, batch_size=BATCH_SIZE * 2)
    if tk.hvd.is_master():
        # スコア表示
        score = compute_score(np.int32(val_dataset.y > 127), np.int32(pred_val > 0.5))
        logger.info(f'score:     {score:.3f}')
        # オレオレ指標
        print_metrics(val_dataset.y > 127, pred_val > 0.5, print_fn=logger.info)
    tk.hvd.barrier()


def load_data(data_dir):
    """データの読み込み。"""
    import pandas as pd

    def _load_image(X):
        X = np.array([tk.ndimage.load(p, grayscale=True) for p in tk.utils.tqdm(X, desc='load')])
        return X

    id_list = pd.read_csv(data_dir / 'train.csv')['id'].values
    X = _load_image([data_dir / 'train' / 'images' / f'{id_}.png' for id_ in id_list])
    y = _load_image([data_dir / 'train' / 'masks' / f'{id_}.png' for id_ in id_list])
    ti, vi = tk.ml.cv_indices(X, y, cv_count=5, cv_index=0, split_seed=6768115, stratify=False)
    (X_train, y_train), (X_val, y_val) = (X[ti], y[ti]), (X[vi], y[vi])

    train_dataset = MyDataset(X_train, y_train, INPUT_SHAPE, data_augmentation=True)
    val_dataset = MyDataset(X_val, y_val, INPUT_SHAPE)
    return train_dataset, val_dataset


def create_model():
    """モデルの作成。"""
    inputs = x = tk.keras.layers.Input(INPUT_SHAPE)
    x = tk.layers.Preprocess(mode='tf')(x)
    x = tk.layers.Pad2D(((5, 6), (5, 6)), mode='reflect')(x)  # 112
    x = tk.keras.layers.concatenate([x, x, x])
    x = tk.layers.CoordChannel2D(x_channel=False)(x)
    x = _conv2d(64, 7, strides=1)(x)  # 112
    x = _conv2d(128, strides=2, use_act=False)(x)  # 56
    x = _blocks(128, 2)(x)
    x = _conv2d(256, strides=2, use_act=False)(x)  # 28
    x = _blocks(256, 4)(x)
    d = x
    x = _conv2d(512, strides=2, use_act=False)(x)  # 14
    x = _blocks(512, 8)(x)
    x = _conv2d(512, strides=2, use_act=False)(x)  # 7
    x = _blocks(512, 4)(x)
    x = _conv2d(128 * 4 * 4)(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 28
    x = _conv2d(256, use_act=False)(x)
    d = _conv2d(256, use_act=False)(d)
    x = tk.keras.layers.add([x, d])
    x = _blocks(256, 3)(x)
    x = _conv2d(1 * 4 * 4, use_act=False)(x)
    x = tk.layers.SubpixelConv2D(scale=4)(x)  # 112
    x = tk.keras.layers.Cropping2D(((5, 6), (5, 6)))(x)  # 101
    logits = x
    x = tk.keras.layers.Activation('sigmoid')(x)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)

    def loss(y_true, y_pred):
        _ = y_pred
        return tk.losses.lovasz_hinge(y_true, logits, from_logits=True)

    base_lr = 1e-3 * BATCH_SIZE * tk.hvd.get().size()
    optimizer = tk.optimizers.NSGD(lr=base_lr, momentum=0.9, nesterov=True, clipnorm=10.0)
    tk.models.compile(model, optimizer, loss, [tk.metrics.binary_accuracy, tk.metrics.binary_iou])
    return model


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


def compute_score(y_true, y_pred):
    """スコア算出。"""
    obj = np.any(y_true, axis=(1, 2, 3))
    empty = np.logical_not(obj)
    pred_empty = np.logical_not(np.any(y_pred, axis=(1, 2, 3)))
    tn = np.logical_and(empty, pred_empty)

    inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)

    prec_list = []
    for threshold in np.arange(0.5, 1.0, 0.05):
        pred_obj = iou > threshold
        match = np.logical_and(obj, pred_obj) + tn
        prec_list.append(np.sum(match) / len(y_true))
    return np.mean(prec_list)


def print_metrics(y_true, y_pred, print_fn):
    """オレオレ指標。"""
    obj = np.any(y_true, axis=(1, 2, 3))
    empty = np.logical_not(obj)

    # 答えが空でないときのIoUの平均
    inter = np.sum(np.logical_and(y_true, y_pred), axis=(1, 2, 3))
    union = np.sum(np.logical_or(y_true, y_pred), axis=(1, 2, 3))
    iou = inter / np.maximum(union, 1)
    iou_mean = np.mean(iou[obj])
    print_fn(f'IoU mean:  {iou_mean:.3f}')

    # 答えが空の場合の正解率
    pred_empty = np.logical_not(np.any(y_pred, axis=(1, 2, 3)))
    acc_empty = np.sum(np.logical_and(empty, pred_empty)) / np.sum(empty)
    print_fn(f'Acc empty: {acc_empty:.3f}')


class MyDataset(tk.data.Dataset):
    """Dataset。"""

    def __init__(self, X, y, input_shape, data_augmentation=False):
        self.X = X
        self.y = y
        self.input_shape = input_shape
        if data_augmentation:
            self.aug = tk.image.Compose([
                tk.image.RandomTransform(width=input_shape[1], height=input_shape[0]),
                tk.image.RandomBlur(p=0.125),
                tk.image.RandomUnsharpMask(p=0.125),
                tk.image.RandomBrightness(p=0.25),
                tk.image.RandomContrast(p=0.25),
            ])
        else:
            self.aug = tk.image.Resize(width=input_shape[1], height=input_shape[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        d = self.aug(image=self.X[index], mask=self.y[index])
        X = d['image']
        y = d['mask'] / 255
        return X, y


if __name__ == '__main__':
    _main()
