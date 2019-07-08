import sys; sys.path.append('.')
import os
import argparse

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from src.utils.engine import train_one_epoch, evaluate
from src.utils import utils
from src.utils.visum_utils import VisumData
from src.utils.transforms import create_transform


def main():
    parser = argparse.ArgumentParser(description='VISUM 2019 competition - baseline training script', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_path', default='/home/master/dataset/train', metavar='', help='data directory path')
    parser.add_argument('-m', '--model_path', default='./baseline.pth', metavar='', help='model file (output of training)')
    parser.add_argument('--epochs', default=50, type=int, metavar='', help='number of epochs')
    parser.add_argument('--lr', default=0.005, type=float, metavar='', help='learning rate')
    parser.add_argument('--l2', default=0.0005, type=float, metavar='', help='L-2 regularization')
    parser.add_argument('--checkpoints_path', default='checkpoints', type=str, help='Directory path to save checkpoints')
    args = vars(parser.parse_args())

    if not os.path.isdir(args['checkpoints_path']):
        os.makedirs(args['checkpoints_path'], exist_ok=True)

    full_checkpoint_path = os.path.join(os.getcwd(), args["checkpoints_path"])
    print(f'Checkpoints will be saved to {full_checkpoint_path}')

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
    dataset = VisumData(args['data_path'], modality='rgb', transforms=create_transform(train=True))
    dataset_val = VisumData(args['data_path'], modality='rgb', transforms=create_transform(train=False))

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

    for epoch in range(args['epochs']):
        # train for one epoch, printing every 10 iterations
        epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluator = evaluate(model, data_loader_val, device=device)

        print(f'Saving the model to {args["checkpoints_path"]}/epoch-{epoch}.pth')
        torch.save(model.state_dict(), f'{args["checkpoints_path"]}/epoch-{epoch}.pth')

    torch.save(model, args['model_path'])

if __name__ == '__main__':
    main()
