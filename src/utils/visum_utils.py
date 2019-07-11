import os
import csv
from typing import Collection, Dict, List, Tuple
from collections import Counter
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import coloredlogs


logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


class VisumData(Dataset):
    # During training our classes have the following idx
    # and evaluation just do not know about background class
    original_class_names = {
        -1: 'n.a.', 0: 'book', 1: 'bottle', 2: 'box',
        3: 'cellphone', 4: 'cosmetics', 5: 'glasses', 6: 'headphones',
        7: 'keys', 8: 'wallet', 9: 'watch',
    }

    def __init__(self, path, modality='rgb', mode='train', transforms=None,
                 excluded_classes:Collection[int]=()):

        self.path = path
        self.transforms = transforms
        self.mode = mode

        assert modality in ['rgb', 'nir', 'all'], \
            'modality should be on of the following: \'rgb\', \'nir\', \'all\''
        self.modality = modality

        if self.mode == 'train':
            self.annotations = dict()

            with open(os.path.join(self.path, 'annotation.csv')) as csv_file:
                for row in csv.reader(csv_file, delimiter=','):
                    if int(row[5]) in excluded_classes: continue

                    file_name = row[0]
                    obj = [float(value) for value in row[1:5]]
                    obj.append(int(row[5]))

                    if file_name in self.annotations:
                        self.annotations[file_name].append(obj)
                    else:
                        self.annotations[file_name] = [obj]

            # Here we keep all the files which we are allowed to use for training
            self.allowed_files_idx = set(f[3:] for f in self.annotations.keys())

            logger.debug(f'We have the following objects distributions: {self.compute_class_distribution()}')

        self.image_files = [f for f in os.listdir(path) if self.check_file(f)]
        logger.debug(f'Visum Dataset has initialized. It contains {len(self.image_files)} images')

    def check_file(self, file_name:str) -> bool:
        if self.modality in ['rgb', 'nir']:
            # load only RGB or NIR images
            return ('.jpg' in file_name) and (self.modality.upper() in file_name) and self.check_file_class(file_name)
        elif self.modality == 'all':
            # load all images (RGB and NIR)
            return ('.jpg' in file_name) and self.check_file_class(file_name)
        else:
            raise NotImplementedError('Unknown modality')

    def check_file_class(self, file_name:str) -> bool:
        """
            Checks, if we can use this file or if it was excluded from training
            (i.e. will be regarded as a new class during testing)
        """
        if self.mode == 'train':
            return file_name[3:] in self.allowed_files_idx
        else:
            return True # accept all classes

    def compute_class_distribution(self):
        counts = Counter(obj[4] for objects in self.annotations.values() for obj in objects)
        freqs = sorted(counts.most_common())

        return freqs

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img = Image.open(os.path.join(self.path, file_name))

        if self.mode == 'train':
            ann_key = self.image_files[idx].replace('NIR', 'RGB')
            ann = self.annotations.get(ann_key, [])

            num_objs = len(ann)
            boxes = list()
            labels = list()

            for ii in range(num_objs):
                boxes.append(ann[ii][0:4])
                labels.append(ann[ii][4])

            if num_objs > 0:
                image_id = torch.tensor([idx])
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

                target = {}
                target["image_id"] = image_id
                target["boxes"] = boxes
                target["area"] = area
                target["iscrowd"] = iscrowd

                # We do +1 here, because 0 class is reserved for background
                # and our annotations do not know about that
                target["labels"] = labels + 1

                assert torch.all(target['labels'] != 0).item(), \
                    "You have provided -1 class for training, dude. Why?"
            else:
                target = None
        else:
            target = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, file_name

    def __len__(self):
        return len(self.image_files)
