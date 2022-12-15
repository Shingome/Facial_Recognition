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


def show_element(el):
    x = el[1][0][::2]
    y = el[1][0][1::2]
    image = np.reshape(el[0], (280, 280, 3))
    plt.imshow(image.astype(int))
    plt.scatter(x, y)
    plt.show()


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


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')

    tf.config.list_physical_devices("GPU")

    train_ds = get_dataset(0)
    val_ds = get_dataset(1)

    train_ds.cache()
    train_ds.prefetch(tf.data.AUTOTUNE)

    model = keras.Sequential()
    model.add(InputLayer((280, 280, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(28, activation='linear'))

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    plot_model(model,
               to_file="models/plot_ep50_aug_8.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=50,
                        shuffle=True,
                        batch_size=32)

    model.save("models/train_ep50_aug_8.h5", save_format="h5")

    pd.DataFrame(history.history).plot()

    plt.show()
