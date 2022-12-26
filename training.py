from keras import models
import os
import numpy as np
import pandas as pd
from keras.losses import MeanSquaredError
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense,\
    BatchNormalization, concatenate, Input, InputLayer
from keras.utils.vis_utils import plot_model
from keras.utils import image_dataset_from_directory
from keras.models import Model
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.ensemble import AdaBoostRegressor


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


def create_cnn(inp):
    x = InputLayer((280, 280,  3))(inp)
    x = BatchNormalization()(x)
    x = Conv2D(16, (6, 6), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (4, 4), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (2, 2), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(1024, (1, 1), activation='relu')(x)
    x = MaxPooling2D((2, 2), 2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    return x



def create_two_brunch():
    def create_part(first, kernel_size):
        inp = first
        part = Conv2D(32, kernel_size, activation='relu')(inp)
        part = MaxPooling2D(pool_size=kernel_size)(part)
        part = BatchNormalization()(part)
        part = Conv2D(64, kernel_size, activation='relu')(part)
        part = MaxPooling2D(pool_size=kernel_size)(part)
        part = BatchNormalization()(part)
        part = Conv2D(128, kernel_size, activation='relu')(part)
        part = MaxPooling2D(pool_size=kernel_size)(part)
        part = BatchNormalization()(part)
        part = Dropout(0.25)(part)
        return part

    inp = Input((280, 280, 3))

    part_1 = create_part(inp, (3, 3))
    part_2 = create_part(inp, (3, 3))

    output = concatenate([part_1, part_2])
    output = Conv2D(512, (2, 2), activation='relu')(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = BatchNormalization()(output)
    output = Conv2D(1024, (2, 2), activation='relu')(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)
    output = Dropout(0.25)(output)
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = Dense(1024, activation='relu')(output)
    output = Dense(28, activation='linear')(output)

    model = Model(inputs=[inp], outputs=[output])
    return model

def create_three_brunch():
    def create_brunch(input_layer):
        inp = input_layer
        part = Conv2D(16, (5, 5), activation='relu')(inp)
        part = MaxPooling2D(pool_size=(2, 2))(part)
        part = BatchNormalization()(part)
        part = Conv2D(32, (3, 3), activation='relu')(part)
        part = MaxPooling2D(pool_size=(2, 2))(part)
        part = BatchNormalization()(part)
        part = Dropout(0.1)(part)
        return part

    inp = Input((280, 280, 3))

    brunch_1 = create_brunch(inp)
    brunch_2 = create_brunch(inp)
    brunch_3 = create_brunch(inp)

    output = concatenate([brunch_1, brunch_2, brunch_3])
    output = Flatten()(output)
    output = Dense(28, activation='linear')(output)

    model = Model(inputs=[inp], outputs=[output])
    return model


def create_three_14_brunch():
    def L1(inp):
        def branch(inp):
            part = Conv2D(16, (5, 5), activation='relu')(inp)
            part = MaxPooling2D(pool_size=(2, 2))(part)
            part = BatchNormalization()(part)
            part = Conv2D(32, (3, 3), activation='relu')(part)
            part = MaxPooling2D(pool_size=(2, 2))(part)
            part = BatchNormalization()(part)
            part = Dropout(0.1)(part)
            return part

        branches = list(branch(inp) for i in range(3))

        out = concatenate(branches)

        return L2(out)

    def L2(inp):
        def two_branch(inp):
            def branch(inp):
                part = Conv2D(64, (3, 3), activation='relu')(inp)
                part = MaxPooling2D(pool_size=(2, 2))(part)
                part = BatchNormalization()(part)
                part = Conv2D(128, (3, 3), activation='relu')(part)
                part = MaxPooling2D(pool_size=(2, 2))(part)
                part = BatchNormalization()(part)
                part = Dropout(0.1)(part)
                return part

            branches = list(branch(inp) for i in range(2))

            out = concatenate(branches)

            return out

        branches = list(two_branch(inp) for i in range(7))

        out = concatenate(branches)

        return L3(out)

    def L3(inp):
        def two_branch(inp):
            def branch(inp):
                part = Conv2D(128, (3, 3), activation='relu')(inp)
                part = MaxPooling2D(pool_size=(2, 2))(part)
                part = BatchNormalization()(part)
                part = Conv2D(128, (3, 3), activation='relu')(part)
                part = MaxPooling2D(pool_size=(3, 3))(part)
                part = BatchNormalization()(part)
                part = Dropout(0.1)(part)
                return part

            branches = list(branch(inp) for i in range(2))

            out = concatenate(branches)

            return out

        branches = list(two_branch(inp) for i in range(7))

        out = concatenate(branches)

        return out

    inp = Input((280, 280, 3))

    output = L1(inp)
    output = Flatten()(output)
    output = Dense(2048, activation='relu')(output)
    output = Dense(28, activation='linear')(output)

    model = Model(inputs=[inp], outputs=[output])
    return model


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(gpu := tf.config.list_physical_devices("GPU"))

    tf.config.experimental.set_memory_growth(gpu[0], True)

    tf.keras.backend.clear_session()

    train_ds = get_dataset(0)
    val_ds = get_dataset(1)

    train_ds.cache()
    train_ds.prefetch(tf.data.AUTOTUNE)

    model = create_simple_model()

    model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['accuracy'])

    # model = models.load_model("models/train_20_4.h5")

    plot_model(model,
               to_file="models/plot_40_10 .png",
               show_dtype=True,
               show_shapes=True,
               show_layer_names=True,
               show_layer_activations=True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=40,
                        shuffle=True,
                        batch_size=32)

    model.save("models/train_40_10.h5", save_format="h5")

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
