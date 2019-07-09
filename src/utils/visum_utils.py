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
    def __init__(self, path, modality='rgb', mode='train', transforms=None,
                 exclude_classes:Collection[int]=()):

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
                    if int(row[5]) in exclude_classes: continue

                    file_name = row[0]
                    obj = [float(value) for value in row[1:5]]
                    obj.append(int(row[5]))

                    if file_name in self.annotations:
                        self.annotations[file_name].append(obj)
                    else:
                        self.annotations[file_name] = [obj]

            self.class_names = {
                0: 'book', 1: 'bottle', 2: 'box', 3: 'cellphone',
                4: 'cosmetics', 5: 'glasses', 6: 'headphones', 7: 'keys',
                8: 'wallet', 9: 'watch', -1: 'n.a.'}

        allowed_files_idx = set(f[3:] for f in self.annotations.keys())

        def check_file(file_name):
            if self.modality in ['rgb', 'nir']:
                # load only RGB or NIR images
                return ('.jpg' in file_name) and (self.modality.upper() in file_name) and (file_name[3:] in allowed_files_idx)
            elif self.modality == 'all':
                # load all images (RGB and NIR)
                return ('.jpg' in file_name) and (file_name[3:] in allowed_files_idx)
            else:
                raise NotImplementedError('Should never happen')

        self.image_files = [f for f in os.listdir(path) if check_file(f)]

        logger.debug(f'Visum Dataset has initialized. It contains {len(self.image_files)} images')
        logger.debug(f'We have the following objects distributions: {self.compute_class_distribution()}')

        if len(exclude_classes) != 0:
            old_class_indices = [i for i in range(10) if not i in exclude_classes]
            new_class_indices = list(range(len(old_class_indices)))
            idx_remap = {old_idx: new_idx for new_idx, old_idx in zip(new_class_indices, old_class_indices)}
            self.annotations = {f: self.remap_objects(self.annotations[f], idx_remap) for f in self.annotations}
            logger.debug(f'Class indices were remapped, because some of them are exluded: {old_class_indices} -> {new_class_indices}')
            logger.debug(f'Now we have the following class distribution: {self.compute_class_distribution()}')

    def compute_class_distribution(self):
        counts = Counter(obj[4] for objects in self.annotations.values() for obj in objects)
        freqs = sorted(counts.most_common())

        return freqs

    def remap_objects(self, objects_list:List[Tuple[float, float, float, float, int]], idx_remap:Dict[int, int]):
        return [obj[:4] + [idx_remap[obj[4]]] for obj in objects_list]

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
                target["labels"] = labels
                target["area"] = area
                target["iscrowd"] = iscrowd
            else:
                target = None
        else:
            target = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, file_name

    def __len__(self):
        return len(self.image_files)
