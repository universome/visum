import argparse
from typing import List, Dict, Tuple, Collection
import pandas as pd
from collections import OrderedDict


def generate_annotations_for_excluded_classes(original_annotations_path:str, result_path:str, exclude_classes:List[int]):
    # The logic is simple: just remap all excluded classes to -1
    # And remap non-excluded classes to their class indicies, used during training
    annotations = read_annotations(original_annotations_path)
    # result = annotations[~annotations.class_idx.isin(exclude_classes)]
    idx_remap = get_idx_remap(exclude_classes)
    annotations['class_idx'] = annotations['class_idx'].map(lambda x: idx_remap[x])
    annotations.to_csv(result_path, index=None, header=None)
    print('Done!')
    print(annotations.head())


def read_annotations(path:str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class_idx'])


def parse_args():
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--original_annotations_path', type=str, default='/home/master/dataset/train/annotation.csv', help='Path to original annotations')
    parser.add_argument('--result_path', type=str, required=True, help='Path where to save results')
    parser.add_argument('--exclude_classes', required=True, type=int, nargs='+', help='Classes to exclude')
    args = vars(parser.parse_args())

    return args


def remap_classes(objects:List[Tuple[float, float, float, float, int]], idx_remap:List[int]):
    return [obj[:4] + [idx_remap[obj[4]]] for obj in objects]


def get_idx_remap(exclude_classes:Collection[int]) -> List[int]:
    new_class_idx = list(range(10 - len(exclude_classes)))
    idx_remap = [(-1 if i in exclude_classes else new_class_idx.pop(0)) for i in range(10)]

    return idx_remap


def main():
    args = parse_args()
    generate_annotations_for_excluded_classes(args['original_annotations_path'], args['result_path'], args['exclude_classes'])


if __name__ == "__main__":
    main()
