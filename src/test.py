import sys; sys.path.extend(['.'])
import os
import argparse
import csv

import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from tqdm import tqdm

from src.utils.nms import nms
from src.utils import utils
from src.utils.transforms import create_transform
from src.utils.engine import train_one_epoch, evaluate
from src.utils.visum_utils import VisumData
from src.train import build_model


def main():
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Loading model
    model = build_model().to(device)
    model.load_state_dict(torch.load(args['model_path']))

    # Loading data
    test_data = VisumData(args['data_path'], 'rgb', mode='test', transforms=create_transform())

    NMS_THR = args['nms_threshold']
    REJECT_THR = args['reject_threshold']

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    predictions = []
    predictions_for_extra_reject_thresholds = [[] for _ in args['extra_reject_thresholds']]

    for i, (imgs, _, file_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # set the model to evaluation mode
        model.eval()

        with torch.no_grad():
            prediction = model(list(img.to(device) for img in imgs))

        boxes = np.array(prediction[0]['boxes'].cpu())
        labels = list(prediction[0]['labels'].cpu())
        scores = list(prediction[0]['scores'].cpu())

        # We were training the model for 11 classes (including background)
        # and the evaluation thing expects our model to have only 10 classes
        # because it does not care about background class
        labels = [l - 1 for l in labels]

        nms_boxes, nms_labels, nms_scores = nms(boxes, labels, scores, NMS_THR)

        predictions.extend(extract_predictions(file_names, nms_boxes, nms_labels, nms_scores, REJECT_THR))

        for i, reject_thresh in enumerate(args['extra_reject_thresholds']):
            predictions_for_extra_reject_thresholds[i].extend(
                extract_predictions(file_names, nms_boxes, nms_labels, nms_scores, reject_thresh))

    with open(args['output'], 'w') as f:
        for pred in predictions:
            f.write("{},{},{},{},{},{},{}\n".format(pred[0], float(pred[1]), float(pred[2]), float(pred[3]), float(pred[4]), int(pred[5]), float(pred[6])))

    for i, reject_thresh in enumerate(args['extra_reject_thresholds']):
        curr_preds_filename = f"{args['output'][:-4]}_rej_thresh_{reject_thresh}.csv"

        with open(curr_preds_filename, 'w') as f:
            for pred in predictions_for_extra_reject_thresholds[i]:
                f.write("{},{},{},{},{},{},{}\n".format(pred[0], float(pred[1]), float(pred[2]), float(pred[3]), float(pred[4]), int(pred[5]), float(pred[6])))


def extract_predictions(file_names, nms_boxes, nms_labels, nms_scores, reject_threshold:float):
    preds = []

    for bb in range(len(nms_labels)):
        pred = np.concatenate((list(file_names), list(nms_boxes[bb, :])))  # bounding box

        if nms_scores[bb] >= reject_threshold:
            pred = np.concatenate((pred, [nms_labels[bb]]))  # object label
        else:
            pred = np.concatenate((pred, [-1]))  # Rejects to classify

        pred = np.concatenate((pred, [nms_scores[bb]]))  # BEST CLASS SCORE
        pred = list(pred)

        preds.append(pred)

    return preds


def parse_args():
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline inference script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--data_path', default='/home/master/dataset/test', metavar='', help='test data directory path')
    parser.add_argument('-m', '--model_path', default='./model.pth', metavar='', help='model file')
    parser.add_argument('-o', '--output', default='./predictions.csv', metavar='', help='output CSV file name')
    parser.add_argument('--nms_threshold', type=float, default=0.1, help="Non Maximum Suppresion threshold")
    parser.add_argument('--reject_threshold', type=float, default=0.6,
        help="Rejection threshold to classify as unknown class (naive approach!)")
    parser.add_argument('--extra_reject_thresholds', type=float, default=[], nargs='+',
        help="If you are optimizing for different thresholds, you can generate many predictions files")

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    main()
