#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import skimage.exposure
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.densenet import densenet121

from utils import CameraDataset, RNG
from optimizers import ClassificationOptimizer

# TODO: implement `that` fine-tuning


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        orig_model = densenet121(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Linear(1024, num_classes)
        nn.init.kaiming_uniform(self.classifier.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=9).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, optimizer, **kwargs):
    # load training data
    print 'Loading and splitting data ...'
    dataset = CameraDataset(kwargs['data_path'], train=True, lazy=not kwargs['not_lazy'])

    # define train and val transforms
    rng = RNG()
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][rng.rand() < 0.5]),
        # TODO: transforms.Lambda(lambda img: skimage.exposure.adjust_gamma(img, gamma=0.8-1.2)),
        # TODO: random jpg compression (70-100)
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # split into train, val in stratified fashion
    sss = StratifiedShuffleSplit(n_splits=1, test_size=kwargs['n_val'],
                                 random_state=kwargs['random_seed'])
    train_index, val_index = list(sss.split(dataset.X, dataset.y))[0]
    train_dataset = CameraDataset(kwargs['data_path'], train=True, lazy=True, transform=train_transform)
    val_dataset   = CameraDataset(kwargs['data_path'], train=True, lazy=True, transform=val_transform)
    train_dataset.X = [dataset.X[i] for i in train_index]
    train_dataset.y = [dataset.y[i] for i in train_index]
    val_dataset.X = [dataset.X[i] for i in val_index]
    val_dataset.y = [dataset.y[i] for i in val_index]

    # define loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=kwargs['batch_size'],
                              shuffle=False,
                              num_workers=3)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=3)

    # freeze features for the first epoch
    for param in model.features.parameters():
        param.requires_grad = False

    max_epoch = optimizer.max_epoch
    optimizer.max_epoch = optimizer.epoch + 1
    optimizer.train(train_loader, val_loader)

    # now unfreeze features
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer.max_epoch = max_epoch
    optimizer.train(train_loader, val_loader)

def predict(optimizer, **kwargs):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = CameraDataset(kwargs['data_path'], train=False, lazy=False,
                            transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=kwargs['batch_size'],
                             shuffle=False,
                             num_workers=3)
    print optimizer.test(test_loader)

def main(**kwargs):
    # build model
    print 'Building model ...'
    model = DenseNet121()
    model_params = [
        {'params': model.features.parameters(), 'lr': kwargs['lr'][0]},
        {'params': model.classifier.parameters(), 'lr': kwargs['lr'][min(1, len(kwargs['lr']) - 1)]},
    ]
    path_template = os.path.join(kwargs['model_dirpath'], '{acc:.4f}-{epoch}')
    optimizer = ClassificationOptimizer(model=model, model_params=model_params,
                                        optim=torch.optim.SGD, optim_params=dict(momentum=0.9),
                                        max_epoch=0, path_template=path_template)

    if kwargs['predict_from']:
        optimizer.load(kwargs['predict_from'])
        predict(optimizer, **kwargs)

    if kwargs['resume_from']:
        print 'Resuming from checkpoint ...'
        optimizer.load(kwargs['resume_from'])
        for param_group in optimizer.optim.param_groups:
            param_group['lr'] *= kwargs['lrm']

    print 'Starting training ...'
    optimizer.max_epoch = optimizer.epoch + kwargs['epochs']
    train(model, optimizer, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, default='data/', metavar='PATH',
                        help='directory for storing augmented data etc.')
    parser.add_argument('--not-lazy', action='store_true',
                        help='if enabled, load all training data into RAM')
    parser.add_argument('--n-val', type=int, default=300, metavar='NV',
                        help='number of validation examples to use')
    parser.add_argument('--batch-size', type=int, default=10, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=[1e-4, 1e-3], metavar='LR', nargs='+',
                        help='initial learning rate(s)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs per unique data')
    parser.add_argument('--lrm', type=float, default=1., metavar='M',
                        help='learning rates multiplier, used only when resume training')
    parser.add_argument('--random-seed', type=int, default=1337, metavar='N',
                        help='random seed for train-val split')
    parser.add_argument('--model-dirpath', type=str, default='models/', metavar='DIRPATH',
                        help='directory path to save the model and predictions')
    parser.add_argument('--resume-from', type=str, default=None, metavar='PATH',
                        help='checkpoint path to resume training from')
    parser.add_argument('--predict-from', type=str, default=None, metavar='PATH',
                        help='checkpoint path to make predictions from')
    main(**vars(parser.parse_args()))
