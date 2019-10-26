#!/usr/bin/env python3
"""imagewoofの実験用コード。(こっちはtrainとvalをひっくり返していない。)

<https://github.com/fastai/imagenette>

## レギュレーション

- No inference time tricks, e.g. no: TTA, validation size > train size
- Must start with random weights
- Must be one of the size/#epoch combinations listed in the table

## 実行結果 (256px/400epochs, LB: 90.2)

val_loss: 1.226
val_acc:  0.910

"""
import functools
import pathlib

import albumentations as A
import tensorflow as tf

import pytoolkit as tk

num_classes = 10
input_shape = (256, 256, 3)
batch_size = 16
data_dir = pathlib.Path(f"data/imagewoof")
models_dir = pathlib.Path(f"models/{pathlib.Path(__file__).stem}")
app = tk.cli.App(output_dir=models_dir)
logger = tk.log.get(__name__)


@app.command(logfile=False)
def check():
    create_model().check()


@app.command(use_horovod=True)
def train():
    train_set, val_set = load_data()
    model = create_model()
    evals = model.train(train_set, val_set)
    tk.notifications.post_evals(evals)


@app.command(use_horovod=True)
def validate(model=None):
    _, val_set = load_data()
    model = create_model().load(models_dir)
    pred = model.predict(val_set)[0]
    if tk.hvd.is_master():
        tk.evaluations.print_classification_metrics(val_set.labels, pred)


def load_data():
    return tk.datasets.load_trainval_folders(data_dir)


def create_model():
    return MyModel(
        train_data_loader=MyDataLoader(data_augmentation=True),
        val_data_loader=MyDataLoader(),
        epochs=400,
        callbacks=[tk.callbacks.CosineAnnealing()],
        models_dir=models_dir,
        model_name_format="model.h5",
        skip_if_exists=False,
        use_horovod=True,
    )


class MyModel(tk.pipeline.KerasModel):
    """KerasModel"""

    def create_network(self) -> tf.keras.models.Model:
        K = tf.keras.backend

        conv2d = functools.partial(
            tf.keras.layers.Conv2D,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )
        bn = functools.partial(
            tf.keras.layers.BatchNormalization,
            gamma_regularizer=tf.keras.regularizers.l2(1e-4),
        )
        act = functools.partial(tf.keras.layers.Activation, "relu")

        def down(filters):
            def layers(x):
                in_filters = K.int_shape(x)[-1]
                g = conv2d(in_filters // 8)(x)
                g = bn()(g)
                g = act()(g)
                g = conv2d(in_filters, use_bias=True, activation="sigmoid")(g)
                x = tf.keras.layers.multiply([x, g])
                x = tf.keras.layers.MaxPooling2D(3, strides=1, padding="same")(x)
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
                    x = tf.keras.layers.add([sc, x])
                x = bn()(x)
                x = act()(x)
                return x

            return layers

        inputs = x = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.concatenate(
            [
                conv2d(16, kernel_size=2, strides=2)(x),
                conv2d(16, kernel_size=4, strides=2)(x),
                conv2d(16, kernel_size=6, strides=2)(x),
                conv2d(16, kernel_size=8, strides=2)(x),
            ]
        )  # 1/2
        x = bn()(x)
        x = act()(x)
        x = tf.keras.layers.concatenate(
            [
                conv2d(64, kernel_size=2, strides=2)(x),
                conv2d(64, kernel_size=4, strides=2)(x),
            ]
        )  # 1/4
        x = bn()(x)
        x = blocks(128, 2)(x)
        x = down(256)(x)  # 1/8
        x = blocks(256, 4)(x)
        x = down(512)(x)  # 1/16
        x = blocks(512, 4)(x)
        x = down(512)(x)  # 1/32
        x = blocks(512, 4)(x)
        x = tk.layers.GeM2D()(x)
        x = tf.keras.layers.Dense(
            num_classes,
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="logits",
        )(x)
        x = tf.keras.layers.Activation(activation="softmax")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        return model

    def create_optimizer(self, mode: str) -> tk.models.OptimizerType:
        del mode
        base_lr = 1e-3 * batch_size * tk.hvd.size()
        optimizer = tf.keras.optimizers.SGD(lr=base_lr, momentum=0.9, nesterov=True)
        return optimizer

    def create_loss(self, model: tf.keras.models.Model) -> tuple:
        def loss(y_true, y_pred):
            del y_pred
            logits = model.get_layer("logits").output
            return tk.losses.categorical_crossentropy(
                y_true, logits, from_logits=True, label_smoothing=0.2
            )

        metrics = ["acc"]
        return loss, metrics


class MyDataLoader(tk.data.DataLoader):
    """DataLoader"""

    def __init__(self, data_augmentation=False):
        super().__init__(
            batch_size=batch_size,
            data_per_sample=2 if data_augmentation else 1,
            parallel=True,
        )
        self.data_augmentation = data_augmentation
        if self.data_augmentation:
            self.aug1 = A.Compose(
                [
                    tk.image.RandomTransform(
                        width=input_shape[1], height=input_shape[0]
                    ),
                    tk.image.RandomColorAugmentors(noisy=True),
                ]
            )
            self.aug2 = tk.image.RandomErasing()
        else:
            self.aug1 = tk.image.Resize(width=input_shape[1], height=input_shape[0])
            self.aug2 = None

    def get_data(self, dataset: tk.data.Dataset, index: int):
        X, y = dataset.get_data(index)
        X = tk.ndimage.load(X)
        X = self.aug1(image=X)["image"]
        y = tf.keras.utils.to_categorical(y, num_classes)
        return X, y

    def get_sample(self, data: list) -> tuple:
        if self.data_augmentation:
            sample1, sample2 = data
            X, y = tk.ndimage.mixup(sample1, sample2, mode="beta")
            X = self.aug2(image=X)["image"]
        else:
            X, y = super().get_sample(data)
        X = tk.ndimage.preprocess_tf(X)
        return X, y


if __name__ == "__main__":
    app.run(default="train")
