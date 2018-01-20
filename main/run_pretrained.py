#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet34, resnet50

import env
from utils import (KaggleCameraDataset, RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot,
                   make_numpy_dataset)
from utils.pytorch_samplers import StratifiedSampler
from optimizers import ClassificationOptimizer


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet121, self).__init__()
        orig_model = densenet121(pretrained=True)
        self.features = nn.Sequential(*list(orig_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.PReLU(),
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


class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        orig_model = resnet34(pretrained=True)
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


def train(optimizer, **kwargs):
    # load training data
    print 'Loading data ...'
    X = np.load(os.path.join(kwargs['data_path'], 'X_folds.npy'))  # (n_folds, *, H, W, C)
    y = np.load(os.path.join(kwargs['data_path'], 'y_folds.npy'))  # (n_folds, *)

    # split into train, val
    fold_index = kwargs['fold']
    X_val = X[fold_index]
    y_val = y[fold_index]
    _, _, H, W, C = X.shape
    X_train = X[np.arange(5) != fold_index].transpose((1, 0, 2, 3, 4)).reshape((-1, H, W, C))
    y_train = y[np.arange(5) != fold_index].T.reshape(-1)

    rng = RNG()
    # noinspection PyTypeChecker
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.uniform(0.8, 1.25))),
        transforms.Lambda(lambda img: jpg_compress(img, quality=rng.randint(70, 100 + 1))),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = make_numpy_dataset(X_train, y_train, train_transform)
    val_dataset = make_numpy_dataset(X_val, y_val, val_transform)

    # define loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=kwargs['batch_size'],
                              shuffle=False,
                              num_workers=4,
                              sampler=StratifiedSampler(class_vector=y_train,
                                                        batch_size=kwargs['batch_size']))
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=4)

    if not kwargs['resume_from']:
        # freeze features for the first epoch
        for param in optimizer.optim.param_groups[0]['params']:
            param.requires_grad = False

        max_epoch = optimizer.max_epoch
        optimizer.max_epoch = optimizer.epoch + 1
        optimizer.train(train_loader, val_loader)

        # now unfreeze features
        for param in optimizer.optim.param_groups[0]['params']:
            param.requires_grad = True

        optimizer.max_epoch = max_epoch
    optimizer.train(train_loader, val_loader)

def predict(optimizer, **kwargs):
    # TTA transform
    test_transform = transforms.Compose([
        transforms.CenterCrop(kwargs['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    rng = RNG(seed=1337)
    base_transform = transforms.Compose([
        transforms.RandomCrop(kwargs['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.uniform(0.8, 1.25))),
        transforms.Lambda(lambda img: jpg_compress(img, quality=rng.randint(70, 100 + 1))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tta_n = 10
    def tta_f(img, n=tta_n - 1):
        out = [test_transform(img)]
        for _ in xrange(n):
            out.append(base_transform(img))
        return torch.stack(out, 0)
    tta_transform = transforms.Compose([
        transforms.Lambda(lambda img: tta_f(img)),
    ])

    # load data
    test_dataset = KaggleCameraDataset(kwargs['data_path'], train=False, lazy=not kwargs['not_lazy'],
                                       transform=tta_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=kwargs['batch_size'],
                             shuffle=False,
                             num_workers=4)

    # compute predictions
    logits, _ = optimizer.test(test_loader)

    # compute and save raw probs
    logits = np.vstack(logits)
    proba = softmax(logits)

    # group and average predictions
    """
    Example
    -------
    >>> P = .01 * (np.arange(24) ** 2).reshape((8, 3))
    >>> P = softmax(P)
    >>> P
    array([[ 0.32777633,  0.33107054,  0.34115313],
           [ 0.30806966,  0.33040724,  0.3615231 ],
           [ 0.28885386,  0.32895498,  0.38219116],
           [ 0.27019182,  0.32672935,  0.40307883],
           [ 0.25213984,  0.32375397,  0.42410619],
           [ 0.23474696,  0.32005991,  0.44519313],
           [ 0.21805443,  0.31568495,  0.46626061],
           [ 0.20209544,  0.31067273,  0.48723183]])
    >>> P.reshape(len(P)/4, 4, 3).mean(axis=1)
    array([[ 0.29872292,  0.32929052,  0.37198656],
           [ 0.22675917,  0.31754289,  0.45569794]])
    """
    proba = proba.reshape(len(proba)/tta_n, tta_n, -1).mean(axis=1)

    fnames = [os.path.split(fname)[-1] for fname in test_dataset.X]
    df = pd.DataFrame(proba)
    df['fname'] = fnames
    df = df[['fname'] + range(10)]
    dirpath = os.path.split(kwargs['predict_from'])[0]
    df.to_csv(os.path.join(dirpath, 'proba.csv'), index=False)

    # compute predictions and save in submission format
    index_pred = unhot(one_hot_decision_function(proba))
    data = {'fname': fnames,
            'camera': [KaggleCameraDataset.target_labels()[int(c)] for c in index_pred]}
    df2 = pd.DataFrame(data, columns=['fname', 'camera'])
    df2.to_csv(os.path.join(dirpath, 'submission.csv'), index=False)


def main(**kwargs):
    # build model
    if not kwargs['model_dirpath'].endswith('/'):
        kwargs['model_dirpath'] += '/'
    print 'Building model ...'
    model = {'densenet121': DenseNet121,
             'resnet34': ResNet34,
             'resnet50': ResNet50
             }[kwargs['model']]()
    model_params = [
        {'params': model.features.parameters(), 'lr': kwargs['lr'][0]},
        {'params': model.classifier.parameters(), 'lr': kwargs['lr'][min(1, len(kwargs['lr']) - 1)]},
    ]
    path_template = os.path.join(kwargs['model_dirpath'], '{acc:.4f}-{epoch}')
    optimizer = ClassificationOptimizer(model=model, model_params=model_params,
                                        optim=torch.optim.SGD, optim_params=dict(momentum=0.9),
                                        loss_func={'logloss': nn.CrossEntropyLoss,
                                                   'hinge': nn.MultiMarginLoss}[kwargs['loss']](),
                                        max_epoch=0, path_template=path_template)

    if kwargs['predict_from']:
        optimizer.load(kwargs['predict_from'])
        predict(optimizer, **kwargs)
        return

    if kwargs['resume_from']:
        print 'Resuming from checkpoint ...'
        optimizer.load(kwargs['resume_from'])
        optimizer.path_template = os.path.join(os.path.split(kwargs['resume_from'])[0], '{acc:.4f}-{epoch}')
        for param_group in optimizer.optim.param_groups:
            param_group['lr'] *= kwargs['lrm']

    print 'Starting training ...'
    optimizer.max_epoch = optimizer.epoch + kwargs['epochs']
    train(optimizer, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, default='../data/', metavar='PATH',
                        help='directory for storing augmented data etc.')
    parser.add_argument('--not-lazy', action='store_true',
                        help='if enabled, load all training data into RAM')
    parser.add_argument('--fold', type=int, default=0, metavar='B',
                        help='which fold to use for validation (0-4)')
    parser.add_argument('--model', type=str, default='densenet121', metavar='PATH',
                        help="model to fine-tune, {'densenet121', 'resnet34', 'resnet50'}")
    parser.add_argument('--loss', type=str, default='logloss', metavar='PATH',
                        help="model to fine-tune, {'logloss', 'hinge'}")
    parser.add_argument('--batch-size', type=int, default=20, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=[1e-4, 1e-3], metavar='LR', nargs='+',
                        help='initial learning rate(s)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs')
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
