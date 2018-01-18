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
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.densenet import densenet121

from utils import (KaggleCameraDataset, RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot)
from utils.pytorch_samplers import StratifiedSampler
from optimizers import ClassificationOptimizer


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
        x = F.avg_pool2d(x, kernel_size=5).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(optimizer, **kwargs):
    # load training data
    print 'Loading and splitting data ...'
    dataset = KaggleCameraDataset(kwargs['data_path'], train=True, lazy=not kwargs['not_lazy'])

    # define train and val transforms
    rng = RNG()
    # noinspection PyTypeChecker
    train_transform = transforms.Compose([
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.uniform(0.8, 1.2))),
        transforms.Lambda(lambda img: jpg_compress(img, quality=rng.randint(70, 100 + 1))),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # split into train, val in stratified fashion
    sss = StratifiedShuffleSplit(n_splits=1, test_size=kwargs['n_val'],
                                 random_state=kwargs['random_seed'])
    train_index, val_index = list(sss.split(dataset.X, dataset.y))[0]
    train_dataset = KaggleCameraDataset(kwargs['data_path'], train=True, lazy=True, transform=train_transform)
    val_dataset   = KaggleCameraDataset(kwargs['data_path'], train=True, lazy=True, transform=val_transform)
    train_dataset.X = [dataset.X[i] for i in train_index]
    train_dataset.y = np.asarray(dataset.y)[train_index]
    val_dataset.X = [dataset.X[i] for i in val_index]
    val_dataset.y = np.asarray(dataset.y)[val_index]

    # define loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=kwargs['batch_size'],
                              shuffle=False,
                              num_workers=3,
                              sampler=StratifiedSampler(class_vector=train_dataset.y,
                                                        batch_size=kwargs['batch_size']))
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=3)

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
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    rng = RNG(seed=1337)
    base_transform = transforms.Compose([
        transforms.RandomCrop(256),
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
                             num_workers=3)

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
    parser.add_argument('--data-path', type=str, default='data/', metavar='PATH',
                        help='directory for storing augmented data etc.')
    parser.add_argument('--not-lazy', action='store_true',
                        help='if enabled, load all training data into RAM')
    parser.add_argument('--n-val', type=int, default=250, metavar='NV',
                        help='number of validation examples to use')
    parser.add_argument('--batch-size', type=int, default=5, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=[1e-4, 1e-3], metavar='LR', nargs='+',
                        help='initial learning rate(s)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
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
