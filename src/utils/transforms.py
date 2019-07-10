from typing import List, Collection

import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import albumentations as A

xmin = 120; xmax = 392; ymin = 291; ymax = 456; epsilon = 20

TRAIN_AUGMENTATIONS = [
    A.HorizontalFlip(0.5),
    A.OneOf([
        A.Blur(p=0.5, blur_limit=5),
        A.MedianBlur(p=0.5, blur_limit=4),
        A.MotionBlur(p=0.5, blur_limit=4),
    ]),
    A.HueSaturationValue(p=0.4, hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15),
    A.RGBShift(p=0.5),
    A.GaussNoise(p=0.5),
    A.CLAHE(p=0.2, tile_grid_size=(8, 8)),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomGamma(p=0.5),
    A.ShiftScaleRotate(p=0.5, shift_limit=0.03, rotate_limit=5, scale_limit=0.05),
    A.JpegCompression(p=0.5),
    A.Crop(x_min=xmin-epsilon, y_min=ymin-epsilon, x_max=xmax+epsilon, y_max=ymax+epsilon, p=1.0)
]


# Data augmentation
def create_transform(augmentations:Collection[str]=None):
    transforms_to_apply = []

    if not augmentations is None:
        albu_transform = A.Compose(augmentations, bbox_params={
            'format': 'pascal_voc',
            'min_area': 0.,
            'min_visibility': 0.,
            'label_fields': ['labels']
        })

        transforms_to_apply.append(albu_transform)

    transforms_to_apply.append(ToTensor())
    return Compose(transforms_to_apply)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            if isinstance(t, A.Compose):
                albu_input = convert_to_albu_format(image, target)
                albu_result = t(**albu_input)
                image, target = convert_from_albu_format(albu_result)
            else:
                image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


def convert_to_albu_format(image, target):
    return {
        "image": np.array(image),
        "bboxes": target["boxes"],
        "labels": target["labels"]
    }


def convert_from_albu_format(albu_result):
    return albu_result["image"], {
        "boxes": torch.Tensor(albu_result["bboxes"]),
        "labels": torch.Tensor(albu_result["labels"])
    }
