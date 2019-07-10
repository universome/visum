import sys; sys.path.extend(['.'])
import os
import random
import argparse
import logging
from datetime import datetime

import torch
import torch.utils.data
import torchvision
import coloredlogs
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from src import detection
from src.utils.engine import train_one_epoch, evaluate
from src.utils import utils
from src.utils.visum_utils import VisumData
from src.utils.transforms import create_transform, TRAIN_AUGMENTATIONS
from src.constants import NUM_CLASSES


logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


def main():
    fix_random_seed(42)

    args = parse_cli_args()
    model = build_model(10 - len(args['exclude_classes']))

    # exclude_classes argument is passed in both training and validation,
    # because validation during training does not bother with "new class" prediction
    dataset = VisumData(args['data_path'], modality='rgb', transforms=create_transform(TRAIN_AUGMENTATIONS), exclude_classes=args['exclude_classes'])
    dataset_val = VisumData(args['data_path'], modality='rgb', transforms=create_transform(), exclude_classes=args['exclude_classes'])

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-args['val_set_size']])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-args['val_set_size']:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args['lr'],
                                momentum=0.9, weight_decay=args['l2'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    tb_writer = SummaryWriter(log_dir=args['log_dir'], flush_secs=10)

    for epoch in range(args['epochs']):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100, tb_writer=tb_writer,
                        accumulation_factor=args['batch_acc'])
        # update the learning rate
        lr_scheduler.step()

        # evaluate the model
        train_evaluator = evaluate(model, data_loader, device=device)
        val_evaluator = evaluate(model, data_loader_val, device=device)
        log_metrics(val_evaluator, tb_writer, epoch, 'val')
        log_metrics(train_evaluator, tb_writer, epoch, 'train')

        logger.info(f'Saving the model to {args["checkpoints_path"]}/epoch-{epoch}.pth')
        torch.save(model.state_dict(), f'{args["checkpoints_path"]}/epoch-{epoch}.pth')

    torch.save(model.state_dict(), args['model_path'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_path', default='/home/master/dataset/train', metavar='', help='data directory path')
    parser.add_argument('-m', '--model_path', default='./baseline.pth', metavar='', help='model file (output of training)')
    parser.add_argument('--epochs', default=50, type=int, metavar='', help='number of epochs')
    parser.add_argument('--lr', default=0.005, type=float, metavar='', help='learning rate')
    parser.add_argument('--l2', default=0.0005, type=float, metavar='', help='L-2 regularization')
    parser.add_argument('--checkpoints_path', default='checkpoints', type=str, help='Directory path to save checkpoints')
    parser.add_argument('--log_dir', type=str, help='Directory where Tensorboard logs are going to be saved', default='tensorboard-logs')
    parser.add_argument('--exclude_classes', type=int, nargs='+', default=[],
        help='Choose, which class idx (0-9) should be excluded during training and added to the validation as a new class.')
    parser.add_argument('--batch-acc', default=1, type=int, metavar='',
                        help='Number of batches accumulated for the parameters update')
    parser.add_argument('--val_set_size', default=100, type=int, help='Number of objects to use in validation')

    args = vars(parser.parse_args())

    exp_name = f'exp_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    logger.info(f'Current experiment name is {exp_name}')
    args['checkpoints_path'] = os.path.join(os.getcwd(), args["checkpoints_path"], exp_name)
    args['log_dir'] = os.path.join(os.getcwd(), args["log_dir"], exp_name)

    if not os.path.isdir(args['checkpoints_path']):
        logger.warn(f'Creating checkpoint directory: {args["checkpoints_path"]}')
        os.makedirs(args['checkpoints_path'], exist_ok=True)

    if not os.path.isdir(args['log_dir']):
        logger.warn(f'Creating tensorboard logs directory: {args["log_dir"]}')
        os.makedirs(args['log_dir'], exist_ok=True)

    logger.info(f'Checkpoints will be saved to {args["checkpoints_path"]}')
    logger.info(f'Tensorboard logs will be save to {args["log_dir"]}')

    return args


def build_model():
    model = detection.fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES, pretrained_backbone=True)

    return model


def log_metrics(coco_evaluator, tb_writer, epoch, mode:str):
    stats = coco_evaluator.coco_eval['bbox'].stats
    tb_writer.add_scalar("AP__IoU_0_50_0_95__area_all__maxDets_100/{mode}", stats[0], epoch)
    # tb_writer.add_scalar("VAL/AP__IoU_0_50__area_all__maxDets_100", stats[1], epoch)
    # tb_writer.add_scalar("VAL/AP__IoU_0_75__area_all__maxDets_100", stats[2], epoch)
    # tb_writer.add_scalar("VAL/AP__IoU_0_50_0_95__area_small__maxDets_100", stats[3], epoch)
    # tb_writer.add_scalar("VAL/AP__IoU_0_50_0_95__area_medium__maxDets_100", stats[4], epoch)
    # tb_writer.add_scalar("VAL/AP__IoU_0_50_0_95__area_ large__maxDets_100", stats[5], epoch)
    # tb_writer.add_scalar("VAL/AR__IoU_0_50_0_95__area_all__maxDets_  1", stats[6], epoch)
    # tb_writer.add_scalar("VAL/AR__IoU_0_50_0_95__area_all__maxDets_ 10", stats[7], epoch)
    # tb_writer.add_scalar("VAL/AR__IoU_0_50_0_95__area_all__maxDets_100", stats[8], epoch)
    # tb_writer.add_scalar("VAL/AR__IoU_0_50_0_95__area_small__maxDets_100", stats[9], epoch)
    # tb_writer.add_scalar("VAL/AR__IoU_0_50_0_95__area_medium__maxDets_100", stats[10], epoch)
    # tb_writer.add_scalar("VAL/AR__IoU_0_50_0_95__area_ large__maxDets_100", stats[11], epoch)


def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()
