#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

import env
from utils import (KaggleCameraDataset, RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot)
from utils.pytorch_samplers import StratifiedSampler
from optimizers import ClassificationOptimizer


class CNN_Small2(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Small2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.PReLU(),
            nn.Linear(2048, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes),
        )
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform(layer.weight.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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
        transforms.RandomCrop(kwargs['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.uniform(0.8, 1.2))),
        transforms.Lambda(lambda img: jpg_compress(img, quality=rng.randint(70, 100 + 1))),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    def aug_f(img, n=kwargs['n_crops']):
        out = [train_transform(img) for _ in xrange(n)]
        return torch.stack(out, 0)
    aug_transform = transforms.Compose([
        transforms.Lambda(lambda img: aug_f(img)),
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(kwargs['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # split into train, val in stratified fashion
    sss = StratifiedShuffleSplit(n_splits=1, test_size=kwargs['n_val'],
                                 random_state=kwargs['random_seed'])
    train_index, val_index = list(sss.split(dataset.X, dataset.y))[0]
    train_dataset = KaggleCameraDataset(kwargs['data_path'], train=True, lazy=True, transform=aug_transform)
    val_dataset = KaggleCameraDataset(kwargs['data_path'], train=True, lazy=True, transform=val_transform)
    train_dataset.X = [dataset.X[i] for i in train_index]
    train_dataset.y = np.asarray(dataset.y)[train_index]
    val_dataset.X = [dataset.X[i] for i in val_index]
    val_dataset.y = np.asarray(dataset.y)[val_index]

    # define loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=kwargs['batch_size'],
                              shuffle=False,
                              num_workers=4,
                              sampler=StratifiedSampler(class_vector=train_dataset.y,
                                                        batch_size=kwargs['batch_size']))
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=4)

    print 'Starting training ...'
    optimizer.train(train_loader, val_loader)

def predict(optimizer, **kwargs):
    # load data
    test_transform = transforms.Compose([
        transforms.CenterCrop(kwargs['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # TTA
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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
    model = CNN_Small2()
    path_template = os.path.join(kwargs['model_dirpath'], '{acc:.4f}-{epoch}')
    optimizer = ClassificationOptimizer(model=model,
                                        optim=torch.optim.SGD, optim_params=dict(lr=kwargs['lr'],
                                                                                 momentum=0.9),
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

    optimizer.max_epoch = optimizer.epoch + kwargs['epochs']
    train(optimizer, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, default='../data/', metavar='PATH',
                        help='directory for storing augmented data etc.')
    parser.add_argument('--not-lazy', action='store_true',
                        help='if enabled, load all training data into RAM')
    parser.add_argument('--n-val', type=int, default=250, metavar='NV',
                        help='number of validation examples to use')
    parser.add_argument('--crop-size', type=int, default=128, metavar='C',
                        help='crop size for patches extracted from training images')
    parser.add_argument('--n-crops', type=int, default=10, metavar='NC',
                        help='how much crops to make from one image during augmentation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate(s)')
    parser.add_argument('--epochs', type=int, default=300, metavar='E',
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
