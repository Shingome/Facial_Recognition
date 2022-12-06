import math
import albumentations as al
import numpy as np
import cv2


def change_keypoits(keypoints):
    x = keypoints[0]
    y = keypoints[1]
    keypoints = np.zeros((28))
    for i in range(0, 28, 2):
        keypoints[i] = x[math.floor(i / 2)]
    for i in range(1, 28, 2):
        keypoints[i] = y[math.floor(i / 2)]
    return keypoints


transform = al.Compose([
    al.Downscale(scale_min=0.1, scale_max=0.25, interpolation=cv2.INTER_AREA, p=0.1),
    al.Equalize(mode='cv', by_channels=True, p=0.1),
    al.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.1),
    al.ISONoise(color_shift=(0.3, 0.5), intensity=(0.3, 0.5), p=0.1),
    al.PixelDropout(dropout_prob=0.03,  p=0.1),
    al.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, p=0.1),
    al.RandomGamma(gamma_limit=(20, 180), p=0.1),
    al.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                  blur_value=7, brightness_coefficient=0.7, p=0.1),
    al.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2,
                     shadow_dimension=5, p=0.1),
    al.RandomSnow(snow_point_lower=0.05, snow_point_upper=0.15, brightness_coeff=2, p=0.1),
    al.HorizontalFlip(p=0.3),
    al.VerticalFlip(p=0.3),
    al.Rotate(p=0.6),
    al.RandomCrop(240, 240, p=0.3),
    al.Resize(280, 280, p=1)],
    keypoint_params=al.KeypointParams(format='xy', remove_invisible=False),
    p=1)

x = np.load("files/x.npy", allow_pickle=True)
y = np.load("files/y.npy", allow_pickle=True)

images = []
values = []

for i in range(25000):
    print(i)
    image = x[i % 5000]
    keypoints = y[i % 5000]
    keypoints = np.asarray((keypoints[::2], keypoints[1::2])).T
    transformed = transform(image=image, keypoints=keypoints)
    image = transformed['image']
    keypoints = np.asarray(transformed['keypoints']).T
    keypoints = change_keypoits(keypoints)
    images.append(image)
    values.append(keypoints)


images = np.asarray(images)
values = np.asarray(values)

np.save("files/x_aug.npy", images, allow_pickle=True)
np.save("files/y_aug.npy", values, allow_pickle=True)
