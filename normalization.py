import pandas as pd
from PIL import Image

train_path = "train.csv"
train = pd.read_csv(train_path)

for i in range(len(train)):
    filename = train["filename"][i]
    image = Image.open("train/" + filename)
    image_normal = image.resize((280, 280))
    k = 280 / image.size[0]
    train.iloc[i, 1:] *= k
    image_normal.save("train_normal/" + filename)

train.to_csv("train_normal.csv", index=False)
