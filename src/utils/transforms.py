import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import albumentations as A
# from albumentations.pytorch import ToTensor


# Data augmentation
def create_transform(train:bool):
    transforms_to_apply = []

    if train:
        augmentations = [
            A.HorizontalFlip(0.5),
            A.Blur(),
            # A.RandomCrop(400, 400),
            A.RandomGamma(),
            A.ShiftScaleRotate(),
            A.HueSaturationValue(),
            A.RGBShift(),
        ]

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
    # TODO: check that if we crop the image too much some labels and bboxes are really removed
    return albu_result["image"], {
        "boxes": torch.Tensor(albu_result["bboxes"]),
        "labels": torch.Tensor(albu_result["labels"])
    }

# class BeforeAlbuTransform(object):
#     def __call__(self, image, target):
#         return {
#             "image": image,
#             "bboxes": target["boxes"],
#             "labels": target["labels"]
#         }
#
# class AfterAlbuTransform(object):
#     def __call__(self, albu_result):
#         return albu_result["image"], {
#             "boxes": albu_result["bboxes"],
#             "labels": albu_result["labels"]
#         }
