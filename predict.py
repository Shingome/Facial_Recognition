import random

import numpy as np
from keras import models
from matplotlib import pyplot as plt
from PIL import Image
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    model = models.load_model("models/test.h5")
    files = glob.glob('test/*.jpg')
    random.shuffle(files)
    for filename in files:
        image = Image.open(filename).resize((96, 96))
        image_arr = np.reshape(np.asarray(image), (1, 96, 96, 3))
        print(predict := model.predict(image_arr))
        x = predict[0, ::2]
        y = predict[0, 1::2]
        plt.imshow(image)
        plt.scatter(x=x, y=y)
        plt.show()
