import pandas as pd
from PIL import Image
import numpy as np
import os
import tensorflow as tf


train = pd.read_csv("files/train_normal.csv")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
images = []
values = []

for i in range(len(train)):
    image = np.asarray(Image.open("train_normal/" + train["filename"][i]))
    value = np.asarray(train.iloc[i, 1:]).astype("float32")
    images.append(image)
    values.append(value)

images = np.asarray(images)
values = np.asarray(values)

np.save("files/x.npy", images, allow_pickle=True)
np.save("files/y.npy", values, allow_pickle=True)
