from typing import List, Collection

import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import albumentations as A


# Data augmentation
def create_transform(augmentations_to_use:Collection[str]=None):
    transforms_to_apply = []

    if not augmentations_to_use is None:
        augmentations = get_augmentations(augmentations_to_use)

        albu_transform = A.Compose(augmentations, bbox_params={
            'format': 'pascal_voc',
            'min_area': 0.,
            'min_visibility': 0.,
            'label_fields': ['labels']
        })

        transforms_to_apply.append(albu_transform)

    transforms_to_apply.append(ToTensor())

    return Compose(transforms_to_apply)


def get_augmentations(augmentations_to_use) -> List[A.Compose]:
    augmentations = []

    if 'HorizontalFlip' in augmentations_to_use:
        augmentations.append(A.HorizontalFlip(0.5))
    if 'Blur' in augmentations_to_use:
        augmentations.append(A.Blur())
    if 'RandomCrop' in augmentations_to_use:
        augmentations.append(A.RandomCrop(400, 400))
    if 'RandomGamma' in augmentations_to_use:
        augmentations.append(A.RandomGamma())
    if 'ShiftScaleRotate' in augmentations_to_use:
        augmentations.append(A.ShiftScaleRotate())
    if 'HueSaturationValue' in augmentations_to_use:
        augmentations.append(A.HueSaturationValue())
    if 'RGBShift' in augmentations_to_use:
        augmentations.append(A.RGBShift())
    if 'RandomSunFlare' in augmentations_to_use:
        augmentations.append(A.RandomSunFlare())

    return augmentations


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
