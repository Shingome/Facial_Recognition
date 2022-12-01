import numpy as np
from keras import models
from matplotlib import pyplot as plt
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    image = Image.open("test_normal/05000.jpg")
    image_arr = np.reshape(np.asarray(image), (1, 280, 280, 3))
    model = models.load_model("models/train_ep5_2.h5")
    predict = model.predict(image_arr)
    x = predict[0, ::2]
    y = predict[0, 1::2]
    plt.imshow(image)
    plt.scatter(x=x, y=y)
    plt.show()
