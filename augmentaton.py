from PIL import Image
import pandas as pd
import math
import albumentations as al
import numpy as np
import cv2
import os
import glob


def change_keypoits(keypoints):
    x = keypoints[0]
    y = keypoints[1]
    keypoints = np.zeros((28))
    for i in range(0, 28, 2):
        keypoints[i] = x[math.floor(i / 2)]
    for i in range(1, 28, 2):
        keypoints[i] = y[math.floor(i / 2)]
    return keypoints


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

transform = al.Compose([
    al.Downscale(scale_min=0.1, scale_max=0.25, interpolation=cv2.INTER_AREA, p=0.1),
    al.Equalize(mode='cv', by_channels=True, p=0.1),
    al.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.1),
    al.ISONoise(color_shift=(0.3, 0.5), intensity=(0.3, 0.5), p=0.1),
    al.PixelDropout(dropout_prob=0.03, p=0.1),
    al.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, p=0.1),
    al.RandomGamma(gamma_limit=(20, 180), p=0.1),
    al.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                  blur_value=7, brightness_coefficient=0.7, p=0.1),
    al.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
                    shadow_dimension=5, p=0.1),
    al.RandomSnow(snow_point_lower=0.05, snow_point_upper=0.1, brightness_coeff=2, p=0.1),
    al.InvertImg(p=0.4),
    al.HorizontalFlip(p=0.5),
    al.VerticalFlip(p=0.5),
    al.Rotate(p=0.7),
    al.RandomCrop(240, 240, p=0),
    al.Resize(280, 280, p=1)],
    keypoint_params=al.KeypointParams(format='xy', remove_invisible=False),
    p=1)

train = pd.read_csv("files/train_normal.csv")
values = []

files = glob.glob('train_aug/1/*')
for f in files:
    os.remove(f)

for i in range(5000):
    print(i)
    filename = train.iloc[i, 0]
    image = Image.open("train_normal/1/" + filename)
    image = image.resize((96, 96))
    image.save("train_aug/1/" + filename)

for i in range(5000, 20000):
    print(i)
    filename = train.iloc[i % 5000, 0]
    keypoints = train.iloc[i % 5000, 1:].to_numpy()
    image = np.asarray(Image.open(f"train_aug/1/{filename}"))
    keypoints = np.asarray((keypoints[::2], keypoints[1::2])).T
    transformed = transform(image=image, keypoints=keypoints)
    keypoints = change_keypoits(np.asarray(transformed['keypoints']).T)
    image = Image.fromarray(transformed['image'], "RGB")
    filename = str(i).zfill(6) + ".jpg"
    values.append(keypoints)
    image.save("train_aug/1/" + filename)

values = np.vstack((train.iloc[::, 1:].to_numpy(), np.asarray(values)))

np.save("files/train_aug_values.npy", values)
