import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

train_path = "train_normal.csv"
train = pd.read_csv(train_path, index_col=False)

for i in range(len(train)):
    filename = train["filename"][i]
    x = train.iloc[i, 1::2]
    y = train.iloc[i, 2::2]
    image = Image.open("train_normal/" + filename)
    plt.imshow(image)
    plt.scatter(x, y)
    plt.show()
