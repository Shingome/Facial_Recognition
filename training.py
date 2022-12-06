import keras
import os
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredError
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":

    tf.get_logger().setLevel('ERROR')

    x = np.load("files/x_aug.npy", allow_pickle=True)
    y = np.load("files/y_aug.npy", allow_pickle=True)

    x_train, x_val = x[:20000], x[20000:]
    y_train, y_val = y[:20000], y[20000:]

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
                        epochs=3,
                        shuffle=True,
                        batch_size=16)

    plot_model(model,
               to_file="models/plot_ep3_aug_1.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    model.save("models/train_ep3_aug_1.h5", save_format="h5")

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.show()

