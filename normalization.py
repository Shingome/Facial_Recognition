import pandas as pd
from PIL import Image

train = pd.read_csv("files/train.csv")

for i in range(len(train)):
    print(i)
    filename = train["filename"][i]
    image = Image.open("train/" + filename).convert("RGB")
    image_normal = image.resize((96, 96))
    k = 96 / image.size[0]
    filename = str(i).zfill(6) + ".jpg"
    train.iloc[i, 0] = filename
    train.iloc[i, 1:] *= k
    image_normal.save("train_normal/1/" + filename)

train.to_csv("files/train_normal.csv", index=False)
