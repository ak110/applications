#!/usr/bin/env python3
"""imagenetteの実験用コード。

[INFO ] val_loss: 1.795
[INFO ] val_acc:  0.840

"""
import functools
import pathlib

import numpy as np

import pytoolkit as tk

num_classes = 10
input_shape = (320, 320, 3)
batch_size = 16
data_dir = pathlib.Path(f"data/imagenette")
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
        epochs=1800,
        callbacks=[tk.callbacks.CosineAnnealing()],
        model_path=models_dir / "model.h5",
    )


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
    if tk.hvd.is_master():
        tk.ml.print_classification_metrics(val_dataset.labels, pred)


@app.command()
@tk.dl.wrap_session(use_horovod=True)
def ml():
    """metric learningお試しコード"""
    train_dataset, val_dataset = load_data()
    model = tk.models.load(models_dir / "model.h5")

    assert isinstance(
        model.layers[-2], tk.keras.layers.GlobalAveragePooling2D
    ), f"layer error: {model.layers[-2]}"
    model = tk.keras.models.Model(model.inputs, model.layers[-2].output)

    train_dataset.data = tk.models.predict(
        model,
        train_dataset,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
    )
    val_dataset.data = tk.models.predict(
        model,
        val_dataset,
        MyPreprocessor(),
        batch_size=batch_size * 2,
        use_horovod=True,
    )
    logger.info("samples_per_class, acc")
    for samples_per_class in [1, 2, 4, 8, 16, 32, 50]:
        ref_dataset = extract(train_dataset, num_classes, samples_per_class)

        import sklearn.metrics.pairwise

        cs = sklearn.metrics.pairwise.cosine_similarity(
            val_dataset.data, ref_dataset.data
        )
        pred = ref_dataset.labels[cs.argmax(axis=-1)]
        acc = np.mean(val_dataset.labels == pred)

        logger.info(f"{samples_per_class}, {acc * 100:.1f}")


def extract(dataset, num_classes, samples_per_class):
    """クラスごとに均等に抜き出す。"""
    index_list = []
    for c in range(num_classes):
        index_list.extend(np.where(dataset.labels == c)[0][:samples_per_class])
    return dataset.slice(index_list)


def load_data():
    class_names, X_train, y_train = tk.ml.listup_classification(data_dir / "train")
    _, X_val, y_val = tk.ml.listup_classification(
        data_dir / "val", class_names=class_names
    )

    # trainとvalを逆にしちゃう。
    (X_train, y_train), (X_val, y_val) = (X_val, y_val), (X_train, y_train)

    return tk.data.Dataset(X_train, y_train), tk.data.Dataset(X_val, y_val)


def create_model():
    conv2d = functools.partial(
        tk.keras.layers.Conv2D,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
        kernel_regularizer=tk.keras.regularizers.l2(1e-4),
    )
    bn = functools.partial(
        tk.keras.layers.BatchNormalization,
        gamma_regularizer=tk.keras.regularizers.l2(1e-4),
    )
    act = functools.partial(tk.keras.layers.Activation, "relu")

    def down(filters):
        def layers(x):
            in_filters = tk.K.int_shape(x)[-1]
            g = conv2d(in_filters // 8)(x)
            g = bn()(g)
            g = act()(g)
            g = conv2d(in_filters, use_bias=True, activation="sigmoid")(g)
            x = tk.keras.layers.multiply([x, g])
            x = tk.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
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
                x = bn(gamma_initializer="zeros")(x)
                x = tk.keras.layers.add([sc, x])
            x = bn()(x)
            x = act()(x)
            return x

        return layers

    inputs = x = tk.keras.layers.Input(input_shape)
    x = tk.keras.layers.concatenate(
        [
            conv2d(16, kernel_size=2, strides=2)(x),
            conv2d(16, kernel_size=4, strides=2)(x),
            conv2d(16, kernel_size=6, strides=2)(x),
            conv2d(16, kernel_size=8, strides=2)(x),
        ]
    )  # 1/2
    x = bn()(x)
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
    logits = tk.keras.layers.Dense(
        num_classes, kernel_regularizer=tk.keras.regularizers.l2(1e-4)
    )(x)
    x = tk.keras.layers.Activation(activation="softmax")(logits)
    model = tk.keras.models.Model(inputs=inputs, outputs=x)
    base_lr = 1e-3 * batch_size * tk.hvd.size()
    optimizer = tk.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)

    def loss(y_true, y_pred):
        del y_pred
        return tk.losses.categorical_crossentropy(
            y_true, logits, from_logits=True, label_smoothing=0.2
        )

    tk.models.compile(model, optimizer, loss, ["acc"])
    return model


class MyPreprocessor(tk.data.Preprocessor):
    """Preprocessor。"""

    def __init__(self, data_augmentation=False):
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug1 = tk.image.Compose(
                [
                    tk.image.RandomTransform(
                        width=input_shape[1], height=input_shape[0]
                    ),
                    tk.image.RandomColorAugmentors(),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        else:
            self.aug1 = tk.image.Resize(width=input_shape[1], height=input_shape[0])
            self.aug2 = None

    def get_sample(
        self, dataset: tk.data.Dataset, index: int, random: np.random.RandomState
    ):
        sample1 = self._get_sample(dataset, index, random)
        if self.data_augmentation:
            sample2 = self._get_sample(dataset, random.choice(len(dataset)), random)
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta", random=random)
            X = self.aug2(image=X, random=random)["image"]
        else:
            X, y = sample1
        X = tk.ndimage.preprocess_tf(X)
        return X, y

    def _get_sample(self, dataset, index, random):
        X, y = dataset.get_sample(index)
        X = tk.ndimage.load(X)
        X = self.aug1(image=X, random=random)["image"]
        y = tk.keras.utils.to_categorical(y, num_classes)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
