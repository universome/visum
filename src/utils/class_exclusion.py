import os
import argparse
import logging
from typing import List, Dict, Tuple, Collection
from collections import OrderedDict

import pandas as pd
import coloredlogs


logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)

"""
To validate on unknown classes you should do the following:
1. train with argument --excluded_classes
2. test with argument --num_excluded_classes
3. run this script by smth like this:
    $ python src/utils/class_exclusion.py --results_dir=class_exclusion_results --excluded_classes 0 1
4. manually evaluate the results for each class by running:
    $ python src/evaluate.py -d class_exclusion_results/class_3 -p class_exclusion_results/class_3/predictions.csv
"""


def generate_annotations_for_excluded_classes(annotations_path:str, preds_path:str, results_dir:str, excluded_classes:List[int]):
    annotations = read_annotations(annotations_path)
    predictions = read_predictions(preds_path)
    idx_remap = get_idx_remap(excluded_classes)
    included_classes = [c for c in range(10) if not c in excluded_classes]

    for c in excluded_classes:
        allowed_classes = included_classes + [c]
        curr_results_dir = os.path.join(results_dir, f'class_{c}')
        os.makedirs(curr_results_dir, exist_ok=True)

        # Generating annotations
        # The logic is simple: just remap all excluded classes to -1
        # And remap non-excluded classes to their class indicies, used during training
        curr_annotations = annotations[annotations.class_idx.isin(allowed_classes)].copy()
        curr_annotations['class_idx'] = curr_annotations.class_idx.map(lambda x: idx_remap[x])
        curr_annotations.to_csv(f'{curr_results_dir}/annotation.csv', index=None, header=None)

        # We need to generate a subset predictions for each excluded class as a new class
        # so we do not have more files in predictions than in annotations
        allowed_files = annotations[annotations.class_idx.isin(allowed_classes)].filename.values
        allowed_files = set(allowed_files)
        curr_predictions = predictions[predictions.filename.isin(allowed_files)].copy()
        curr_predictions.to_csv(f'{curr_results_dir}/predictions.csv', index=None, header=None)

    print('Done!')


def read_annotations(path:str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class_idx'])


def read_predictions(path:str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class_idx', 'confidence'])

def parse_args():
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--annotations_path', type=str, default='/home/master/dataset/train/annotation.csv', help='Path to original annotations')
    parser.add_argument('--predictions_path', type=str, default='./predictions.csv', help='Path to original predictions')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory where to save the results')
    parser.add_argument('--excluded_classes', required=True, type=int, nargs='+', help='Classes to exclude')
    args = vars(parser.parse_args())

    if not os.path.isdir(args['results_dir']):
        logger.warn(f'Creating results_dir directory: {args["results_dir"]}')
        os.makedirs(args['results_dir'], exist_ok=True)

    return args


def remap_classes(objects:List[Tuple[float, float, float, float, int]], idx_remap:List[int]):
    return [obj[:4] + [idx_remap[obj[4]]] for obj in objects]


def get_idx_remap(excluded_classes:Collection[int]) -> List[int]:
    new_class_idx = list(range(10 - len(excluded_classes)))
    idx_remap = [(-1 if i in excluded_classes else new_class_idx.pop(0)) for i in range(10)]

    return idx_remap


def main():
    args = parse_args()
    generate_annotations_for_excluded_classes(args['annotations_path'], args['predictions_path'], args['results_dir'], args['excluded_classes'])


if __name__ == "__main__":
    main()
