import glob
from PIL import Image


def average_size(path="train/*.jpg"):
    x_med = y_med = 0
    path = glob.glob(path)
    for i in path:
        image = Image.open(i)
        x, y = image.size
        x_med += x
        y_med += y

    return x_med / len(path), y_med / len(path)
