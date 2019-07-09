import sys; sys.path.append('.')
import os
import argparse
import logging

import torch
import torch.utils.data
import torchvision
import coloredlogs
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from src.utils.engine import train_one_epoch, evaluate
from src.utils import utils
from src.utils.visum_utils import VisumData
from src.utils.transforms import create_transform


DEFAULT_AUGMENTATIONS = (
    'HorizontalFlip',
    # 'Blur',
    # 'RandomCrop',
    # 'RandomGamma',
    # 'ShiftScaleRotate',
    # 'HueSaturationValue',
    # 'RGBShift',
    # 'RandomSunFlare',
)

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


def main():
    args = parse_cli_args()

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=10,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # See the model architecture
    # print(model)

    # use our dataset and defined transformations
    dataset = VisumData(args['data_path'], modality='rgb', transforms=create_transform(DEFAULT_AUGMENTATIONS))
    dataset_val = VisumData(args['data_path'], modality='rgb', transforms=create_transform())

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-100:])

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
    tb_writer = SummaryWriter(log_dir=args['log_dir'])

    for epoch in range(args['epochs']):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100, tb_writer=tb_writer)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the val dataset
        coco_evaluator = evaluate(model, data_loader_val, device=device)
        log_metrics(coco_evaluator, tb_writer)

        logger.info(f'Saving the model to {args["checkpoints_path"]}/epoch-{epoch}.pth')
        torch.save(model.state_dict(), f'{args["checkpoints_path"]}/epoch-{epoch}.pth')

    torch.save(model, args['model_path'])


def parse_cli_args():
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_path', default='/home/master/dataset/train', metavar='', help='data directory path')
    parser.add_argument('-m', '--model_path', default='./baseline.pth', metavar='', help='model file (output of training)')
    parser.add_argument('--epochs', default=50, type=int, metavar='', help='number of epochs')
    parser.add_argument('--lr', default=0.005, type=float, metavar='', help='learning rate')
    parser.add_argument('--l2', default=0.0005, type=float, metavar='', help='L-2 regularization')
    parser.add_argument('--checkpoints_path', default='checkpoints', type=str, help='Directory path to save checkpoints')
    parser.add_argument('--augmentations', default=DEFAULT_AUGMENTATIONS, type=str, help='List of augmentations names to use', nargs='+')
    parser.add_argument('--log_dir', type=str, help='Directory where Tensorboard logs are going to be saved', default='tensorboard-logs')
    parser.add_argument('--exclude_classes', type=int, nargs='+', default=[],
        help='Choose, which class idx (0-10) should be excluded during training and added to the validation as a new class.')

    args = vars(parser.parse_args())

    if not os.path.isdir(args['checkpoints_path']):
        logger.warn(f'Creating checkpoint directory: {args["checkpoints_path"]}')
        os.makedirs(args['checkpoints_path'], exist_ok=True)

    if not os.path.isdir(args['log_dir']):
        logger.warn(f'Creating checkpoint directory: {args["checkpoints_path"]}')
        os.makedirs(args['log_dir'], exist_ok=True)

    logger.info(f'Checkpoints will be saved to {os.path.join(os.getcwd(), args["checkpoints_path"])}')
    logger.info(f'The following augmentations will be used for training: {args["augmentations"]}')
    logger.info(f'Tensorboard logs will be save to {args["log_dir"]}')

    return args


def log_metrics(coco_evaluator, tb_writer):
    stats = coco_evaluator.coco_eval['bbox'].stats
    tb_writer.add_scalar("VAL/Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", stats[0])
    tb_writer.add_scalar("VAL/Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]", stats[1])
    tb_writer.add_scalar("VAL/Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]", stats[2])
    tb_writer.add_scalar("VAL/Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", stats[3])
    tb_writer.add_scalar("VAL/Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", stats[4])
    tb_writer.add_scalar("VAL/Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", stats[5])
    tb_writer.add_scalar("VAL/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", stats[6])
    tb_writer.add_scalar("VAL/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]", stats[7])
    tb_writer.add_scalar("VAL/Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", stats[8])
    tb_writer.add_scalar("VAL/Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", stats[9])
    tb_writer.add_scalar("VAL/Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", stats[10])
    tb_writer.add_scalar("VAL/Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", stats[11])


if __name__ == '__main__':
    main()
