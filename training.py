import keras
import os
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredError
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer, Rescaling
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    tf.get_logger().setLevel('ERROR')

    dataset_train = tf.data.experimental.load("files/dataset_train")
    dataset_val = tf.data.experimental.load("files/dataset_val")

    print(int(dataset_val.cardinality()))

    exit()

    model = keras.Sequential()
    model.add(InputLayer((280, 280, 3)))
    model.add(Rescaling(1. / 255))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='relu'))

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_val, y_val),
                        epochs=10,
                        shuffle=True,
                        batch_size=32)

    plot_model(model,
               to_file="models/plot_ep10_aug_2.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    model.save("models/train_ep10_aug_2.h5", save_format="h5")

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.show()
