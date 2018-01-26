#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import densenet121, densenet201
from torchvision.models.resnet import resnet34, resnet50, resnet101, resnet152

import env
from optimizers import ClassificationOptimizer
from run import predict, train2


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        orig_model = densenet121(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet201, self).__init__()
        orig_model = densenet201(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(1920, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=5).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        orig_model = resnet34(pretrained=True)
        # self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.features = nn.Sequential(*list(orig_model.children())[:-2])
        # 2048-256-10
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=3) #
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        orig_model = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet101, self).__init__()
        orig_model = resnet101(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet152, self).__init__()
        orig_model = resnet152(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )
        for layer in self.classifier.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=2).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_optimizer_pretrained(optimizer, train_loader, val_loader, **kwargs):
    if optimizer.epoch == 0:
        # freeze features for the first epoch
        for param in optimizer.optim.param_groups[0]['params']:
            param.requires_grad = False

        max_epoch = optimizer.max_epoch
        optimizer.max_epoch = optimizer.epoch + kwargs['epochs_per_unique_data']
        optimizer.train(train_loader, val_loader)

        # now unfreeze features
        for param in optimizer.optim.param_groups[0]['params']:
            param.requires_grad = True

        optimizer.max_epoch = max_epoch
    optimizer.train(train_loader, val_loader)


def train(optimizer, **kwargs):
    # train_loader, val_loader = make_train_loaders(means=(0.485, 0.456, 0.406),
    #                                                stds=(0.229, 0.224, 0.225), **kwargs)
    #
    # train_optimizer_pretrained(optimizer, train_loader, val_loader)
    train2(optimizer, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225),
           train_optimizer=train_optimizer_pretrained, **kwargs)


def main(**kwargs):
    # build model
    if not kwargs['model_dirpath'].endswith('/'):
        kwargs['model_dirpath'] += '/'
    print 'Building model ...'
    model = {'densenet121': DenseNet121,
             'densenet201': DenseNet201,
             'resnet34': ResNet34,
             'resnet50': ResNet50,
             'resnet101': ResNet101,
             'resnet152': ResNet152,
             }[kwargs['model']]()
    model_params = [
        {'params': model.features.parameters(),
         'lr': kwargs['lr'][0]},
        {'params': model.classifier.parameters(),
         'lr': kwargs['lr'][min(1, len(kwargs['lr']) - 1)],
         'weight_decay': 1e-5},
    ]
    path_template = os.path.join(kwargs['model_dirpath'], '{acc:.4f}-{epoch}')
    optimizer = ClassificationOptimizer(model=model, model_params=model_params,
                                        optim=torch.optim.SGD, optim_params=dict(momentum=0.9),
                                        loss_func={'logloss': nn.CrossEntropyLoss,
                                                   'hinge': nn.MultiMarginLoss}[kwargs['loss']](),
                                        max_epoch=0, val_each_epoch=kwargs['epochs_per_unique_data'],
                                        cyclic_lr=kwargs['cyclic_lr'],
                                        path_template=path_template)

    if kwargs['predict_from']:
        optimizer.load(kwargs['predict_from'])
        predict(optimizer,
                means=(0.485, 0.456, 0.406),
                stds=(0.229, 0.224, 0.225),
                **kwargs)
        return

    if kwargs['resume_from']:
        if not kwargs['resume_from'].endswith('ckpt') and not kwargs['resume_from'].endswith('/'):
            kwargs['resume_from'] += '/'
        print 'Resuming from checkpoint ...'
        optimizer.load(kwargs['resume_from'])
        optimizer.path_template = os.path.join(*(list(os.path.split(kwargs['resume_from'])[:-1]) + ['{acc:.4f}-{epoch}']))
        for param_group in optimizer.optim.param_groups:
            param_group['lr'] *= kwargs['lrm']

    print 'Starting training ...'
    optimizer.max_epoch = optimizer.epoch + kwargs['epochs']
    train(optimizer, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, default='../data/', metavar='PATH',
                        help='directory for storing augmented data etc.')
    parser.add_argument('--n-workers', type=int, default=4, metavar='NW',
                        help='how many threads to use for I/O')
    parser.add_argument('--crop-size', type=int, default=256, metavar='C',
                        help='crop size for patches extracted from training images')
    parser.add_argument('--fold', type=int, default=0, metavar='B',
                        help='which fold to use for validation (0-49)')
    parser.add_argument('--n-train-folds', type=int, default=4, metavar='NF',
                        help='number of fold used for training (each is ~880 Mb)')
    parser.add_argument('--skip-train-folds', type=int, default=0, metavar='SF',
                        help='how many folds/blocks to skip at the beginning of training')
    parser.add_argument('--model', type=str, default='densenet121', metavar='PATH',
                        help="model to fine-tune, 'densenet{121, 201}' or 'resnet{34, 50, 101, 152}'")
    parser.add_argument('--loss', type=str, default='logloss', metavar='PATH',
                        help="loss function, {'logloss', 'hinge'}")
    parser.add_argument('--batch-size', type=int, default=20, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=[1e-4, 1e-3], metavar='LR', nargs='+',
                        help='initial learning rate(s)')
    parser.add_argument('--cyclic-lr', type=float, default=None, metavar='CLR', nargs='+',
                        help='cyclic LR in form (lr-min, lr-max, stepsize)')
    parser.add_argument('--epochs', type=int, default=150, metavar='E',
                        help='number of epochs')
    parser.add_argument('--epochs-per-unique-data', type=int, default=8, metavar='EU',
                        help='number of epochs run per unique subset of data')
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
    parser.add_argument('--tta-n', type=int, default=32, metavar='NC',
                        help='number of crops to generate in TTA per test image')
    parser.add_argument('--kernel', action='store_true',
                        help='whether to apply kernel for images prior training')
    main(**vars(parser.parse_args()))
