from typing import List, Collection

import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import albumentations as A

xmin = 120; xmax = 392; ymin = 291; ymax = 456; epsilon = 20


TRAIN_AUGMENTATIONS = [
    # A.Crop(x_min=xmin-epsilon, y_min=ymin-epsilon, x_max=xmax+epsilon, y_max=ymax+epsilon, p=1.0),
    # A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.Blur(p=0.5, blur_limit=3),
        A.MedianBlur(p=0.5, blur_limit=3),
        A.MotionBlur(p=0.5, blur_limit=3),
    ]),
    # A.HueSaturationValue(p=0.4, hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15),
    # A.RGBShift(p=0.5),
    A.GaussNoise(p=0.5, var_limit=(10.0, 30.0)),
    # A.CLAHE(p=0.2, tile_grid_size=(8, 8)),
    A.RandomBrightnessContrast(p=0.3),
    # A.RandomGamma(p=0.5),
    # A.ShiftScaleRotate(p=0.5, shift_limit=0.03, rotate_limit=5, scale_limit=0.05),
    # A.JpegCompression(p=0.5),
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
    transforms_to_apply.append(CenterCrop(xmin, xmax, ymin, ymax, epsilon))

    return Compose(transforms_to_apply)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.to_tensor = ToTensor()

    def __call__(self, image, target):
        for t in self.transforms:
            if isinstance(t, A.Compose):
                albu_input = get_input_for_albumentations(image, target)
                albu_result = t(**albu_input)
                image, target = merge_albu_result_with_target(albu_result, target)
            else:
                image, target = t(image, target)

        return image, target


class CenterCrop(object):
    def __init__(self, xmin, xmax, ymin, ymax, epsilon):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.epsilon = epsilon

    def __call__(self, image, target):
        assert image.size(0) == 3

        image = image[:, ymin - epsilon : ymax + epsilon, xmin - epsilon : xmax + epsilon]

        target['boxes'] = torch.stack([self.center_crop_bbox(bb) for bb in target['boxes']])

        return image, target

    def center_crop_bbox(self, bbox:torch.Tensor):
        return torch.tensor([bbox[0] - xmin, bbox[1] - ymin, bbox[2] - xmin, bbox[3] - ymin])


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


def get_input_for_albumentations(image, target):
    return {
        "image": np.array(image),
        "bboxes": target["boxes"],
        "labels": target["labels"]
    }


def merge_albu_result_with_target(albu_result, target):
    return albu_result["image"], {
        "boxes": torch.tensor(albu_result["bboxes"]),
        "labels": torch.tensor(albu_result["labels"]),
        "image_id": target["image_id"],

        # TODO: actually, area could change after transformations, but let's hope that it didn't
        "area": target["area"]
    }
