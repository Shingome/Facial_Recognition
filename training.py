import keras
import os
from keras.losses import MeanSquaredError
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer
from keras.utils.vis_utils import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == "__main__":

    #Prepare dataset
    #
    #
    #
    #
    #

    model = keras.Sequential()
    model.add(InputLayer((280, 280, 3)))
    model.add(Conv2D(16, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='relu'))

    plot_model(model,
               to_file="model_plot.png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, shuffle=True, batch_size=32)