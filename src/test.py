import sys; sys.path.append('.')
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
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline inference script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_path', default='/home/master/dataset/test', metavar='', help='test data directory path')
    parser.add_argument('-m', '--model_path', default='./baseline.pth', metavar='', help='model file')
    parser.add_argument('-o', '--output', default='./predictions.csv', metavar='', help='output CSV file name')
    args = vars(parser.parse_args())

    NMS_THR = 0.1  # non maximum suppresion threshold
    REJECT_THR = 0.5  # rejection threshold to classify as unknown class (naive approach!)

    # Load datasets
    test_data = VisumData(args['data_path'], 'rgb', mode='test', transforms=create_transform())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = build_model().to(device)
    model.load_state_dict(torch.load(args['model_path']))

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    predictions = list()
    for i, (imgs, _, file_names) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # set the model to evaluation mode
        model.eval()

        with torch.no_grad():
            prediction = model(list(img.to(device) for img in imgs))

        boxes = np.array(prediction[0]['boxes'].cpu())
        labels = list(prediction[0]['labels'].cpu())
        scores = list(prediction[0]['scores'].cpu())

        nms_boxes, nms_labels = nms(boxes, labels, NMS_THR)

        for bb in range(len(nms_labels)):
            pred = np.concatenate((list(file_names), list(nms_boxes[bb, :])))  # bounding box
            if scores[bb] >= REJECT_THR:
                pred = np.concatenate((pred, [nms_labels[bb]]))  # object label
            else:
                pred = np.concatenate((pred, [-1]))  # Rejects to classify
            pred = np.concatenate((pred, [scores[bb]]))  # BEST CLASS SCORE
            pred = list(pred)
            predictions.append(pred)

    with open(args['output'], 'w') as f:
        for pred in predictions:
            f.write("{},{},{},{},{},{},{}\n".format(pred[0], float(pred[1]), float(pred[2]), float(pred[3]), float(pred[4]), int(pred[5]), float(pred[6])))

if __name__ == '__main__':
    main()
