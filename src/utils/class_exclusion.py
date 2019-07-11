import os
import argparse
import logging
from shutil import copyfile
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


def generate_annotations_for_excluded_classes(annotations_path:str, preds_path:str, val_imgs_dir:str, results_dir:str, excluded_classes:List[int]):
    annotations = read_annotations(annotations_path)
    predictions = read_predictions(preds_path)
    val_imgs = [f for f in os.listdir(val_imgs_dir) if f[-4:] == '.jpg']

    if not os.path.isdir(results_dir):
        logger.warn(f'Creating results directory: {results_dir}')
        os.makedirs(results_dir, exist_ok=True)

    # Generating annotations
    # We should save only validation images (and those which are new objects, because they are validation too)
    annotations = annotations[annotations.filename.isin(val_imgs) | annotations.class_idx.isin(excluded_classes)]
    annotations.loc[annotations.class_idx.isin(excluded_classes), 'class_idx'] = -1
    annotations.to_csv(os.path.join(results_dir, 'annotation.csv'), index=None, header=None)

    # Generate a subset of prediction file which corresponds to the current annotation file
    # (i.e. based on the same images)
    allowed_images = set(annotations.filename.values)
    predictions = predictions[predictions.filename.isin(allowed_images)].copy()
    predictions.to_csv(os.path.join(results_dir, 'predictions.csv'), index=None, header=None)

    # evaluation.py works in such a way that it requires a directory of files :|
    for img in allowed_images:
        copyfile(f'/home/master/dataset/train/{img}', f'{results_dir}/{img}')

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
    parser.add_argument('--val_imgs_dir', type=str, help="Dir to validation img", default='/home/visum/mody/models_day3/dataset_seed42')
    args = vars(parser.parse_args())

    if not os.path.isdir(args['results_dir']):
        logger.warn(f'Creating results_dir directory: {args["results_dir"]}')
        os.makedirs(args['results_dir'], exist_ok=True)

    return args


def main():
    args = parse_args()
    generate_annotations_for_excluded_classes(
        args['annotations_path'], args['predictions_path'], args['val_imgs_dir'],
        args['results_dir'], args['excluded_classes'])


if __name__ == "__main__":
    main()
