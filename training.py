import keras
import os
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredError
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer, Rescaling
from keras.utils.vis_utils import plot_model
from keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_dataset(subset):
    train = np.load("files/train_aug_values.npy")
    return image_dataset_from_directory(directory="train_aug",
                                        labels=list(train),
                                        image_size=(280, 280),
                                        batch_size=32,
                                        color_mode="rgb",
                                        validation_split=0.1,
                                        subset=("training", "validation")[subset],
                                        seed=100)


def create_model():
    model = keras.Sequential()
    model.add(InputLayer((280, 280, 3)))
    model.add(Conv2D(16, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(1024, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(28, activation='linear'))
    return model


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    tf.config.list_physical_devices("GPU")

    train_ds = get_dataset(0)
    val_ds = get_dataset(1)

    train_ds.cache()
    train_ds.prefetch(tf.data.AUTOTUNE)

    model = create_model()

    # from keras import models
    # model = models.load_model("models/train_ep40_aug_13_2.h5")

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    plot_model(model,
               to_file="models/plot_ep20_1.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=20,
                        shuffle=True,
                        batch_size=32)

    model.save("models/train_ep20_1.h5", save_format="h5")

    history = pd.DataFrame(history.history)

    figure, axis = plt.subplots(2, 1)

    axis[0].plot(history['loss'], "b-", label="loss")
    axis[0].plot(history['val_loss'], "r-", label="val_loss")
    axis[0].legend()
    axis[1].plot(history['accuracy'], "b-", label="accuracy")
    axis[1].plot(history['val_accuracy'], "r-", label="val_accuracy")
    axis[1].legend()

    plt.show()
