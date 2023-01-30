import os
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredError
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    BatchNormalization, concatenate, Input, InputLayer
from keras.utils.vis_utils import plot_model
from keras.utils import image_dataset_from_directory
from keras.models import Model
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras.applications import mobilenet_v2
from keras.optimizers import Adam, SGD
from numba import cuda
from sklearn.ensemble import AdaBoostRegressor


epochs = 100
shape = (96, 96)
batch = 64
optimazer = SGD(0.01, momentum=0.75, nesterov=True)
optimazer = Adam(0.0005)


def get_dataset(subset):
    train = np.load("files/train_aug_values.npy")
    return image_dataset_from_directory(directory="train_aug",
                                        labels=list(train),
                                        image_size=shape,
                                        batch_size=batch,
                                        color_mode="rgb",
                                        validation_split=0.2,
                                        subset=("training", "validation")[subset],
                                        seed=100)


# def create_cnn(inp):
#     x = Conv2D(32, (5, 5), activation='relu')(inp)
#     x = MaxPooling2D((2, 2))(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(128, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(256, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.25)(x)
#     x = Flatten()(x)
#     x = Dense(7, activation='linear')(x)
#
#     return x

def create_cnn(inp):
    x = Conv2D(32, (3, 3))(inp)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3))(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (2, 2))(x)
    x = MaxPooling2D((2, 2), strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(500, activation='sigmoid')(x)
    x = Dense(500, activation='sigmoid')(x)
    x = Dense(28, activation='linear')(x)
    return x


def create_brunch(num):
    inp = Input((shape[0], shape[1], 3))
    layers = list(create_cnn(inp) for i in range(num))
    out = layers if len(layers) == 1 else concatenate(layers)
    model = Model(inputs=inp, outputs=out)
    return model


# def MobileNet():
#     backbone = keras.applications.MobileNetV2(
#         weights="imagenet", include_top=False, input_shape=(224, 224, 3)
#     )
#     backbone.trainable = False
#
#     inputs = layers.Input((224, 224, 3))
#     x = mobilenet_v2.preprocess_input(inputs)
#     x = backbone(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.SeparableConv2D(28, kernel_size=5, strides=1, activation="relu")(x)
#     outputs = layers.SeparableConv2D(28, kernel_size=3, strides=1, activation="sigmoid")(x)
#
#     return keras.Model(inputs, outputs)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(device := cuda.get_current_device())
    device.reset()

    print(gpu := tf.config.list_physical_devices("GPU"))

    tf.config.experimental.set_memory_growth(gpu[0], True)

    tf.keras.backend.clear_session()

    train_ds = get_dataset(0)
    val_ds = get_dataset(1)

    train_ds.cache()
    train_ds.prefetch(tf.data.AUTOTUNE)

    model = create_brunch(1)

    model.compile(optimizer=optimazer, loss=MeanSquaredError(), metrics=['accuracy'])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/",
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    plot_model(model,
               to_file="models/test.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        shuffle=True,
                        batch_size=batch,
                        callbacks=[model_checkpoint_callback])

    model.save("models/test.h5", save_format="h5")

    ev = model.evaluate(val_ds, verbose=0)

    print("End = " + str(ev))

    history = pd.DataFrame(history.history)

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(history['loss'], "b-", label="loss")
    axis[0].plot(history['val_loss'], "r-", label="val_loss")
    axis[0].legend()
    axis[1].plot(history['accuracy'], "b-", label="accuracy")
    axis[1].plot(history['val_accuracy'], "r-", label="val_accuracy")
    axis[1].legend()

    plt.show()
