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
from models import CNN_Small
from optimizers import ClassificationOptimizer
from utils import (KaggleCameraDataset, RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot,
                   make_numpy_dataset)
from utils.pytorch_samplers import StratifiedSampler


def train(optimizer, **kwargs):
    # load training data
    print 'Loading and splitting data ...'
    if os.path.isfile(os.path.join(kwargs['data_path'], 'X_train.npy')):
        X_train = np.load(os.path.join(kwargs['data_path'], 'X_train.npy'))
        y_train = np.load(os.path.join(kwargs['data_path'], 'y_train.npy'))
        X_val = np.load(os.path.join(kwargs['data_path'], 'X_val.npy'))
        y_val = np.load(os.path.join(kwargs['data_path'], 'y_val.npy'))
    else:
        X = np.load(os.path.join(kwargs['data_path'], 'X_patches.npy'))
        y = np.load(os.path.join(kwargs['data_path'], 'y_patches.npy'))

        # split into train, val in stratified fashion
        sss = StratifiedShuffleSplit(n_splits=1, test_size=kwargs['n_val'],
                                     random_state=kwargs['random_seed'])
        train_ind, val_ind = list(sss.split(np.zeros_like(y), y))[0]
        X_train = X[train_ind]
        y_train = y[train_ind]
        X_val = X[val_ind]
        y_val = y[val_ind]
        np.save(os.path.join(kwargs['data_path'], 'X_train.npy'), X_train)
        np.save(os.path.join(kwargs['data_path'], 'y_train.npy'), y_train)
        np.save(os.path.join(kwargs['data_path'], 'X_val.npy'), X_val)
        np.save(os.path.join(kwargs['data_path'], 'y_val.npy'), y_val)

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

    print 'Starting training ...'
    optimizer.train(train_loader, val_loader)

def predict(optimizer, **kwargs):
    # load data
    X_test = np.load(os.path.join(kwargs['data_path'], 'X_test.npy'))
    y_test = np.zeros((len(X_test),), dtype=np.int64)

    test_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # TTA
    rng = RNG(seed=1337)
    base_transform = transforms.Compose([
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
    tta_n = 10
    def tta_f(img, n=tta_n - 1):
        out = [test_transform(img)]
        for _ in xrange(n):
            out.append(base_transform(img))
        return torch.stack(out, 0)
    tta_transform = transforms.Compose([
        transforms.Lambda(lambda img: tta_f(img)),
    ])

    test_loader = DataLoader(dataset=make_numpy_dataset(X_test, y_test, tta_transform),
                             batch_size=kwargs['batch_size'],
                             shuffle=False,
                             num_workers=4)
    test_dataset = KaggleCameraDataset(kwargs['data_path'], train=False, lazy=not kwargs['not_lazy'])

    # compute predictions
    logits, _ = optimizer.test(test_loader)

    # compute and save raw probs
    logits = np.vstack(logits)
    proba = softmax(logits)

    # group and average predictions
    K = 16 * tta_n
    proba = proba.reshape(len(proba)/K, K, -1).mean(axis=1)

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
    model = CNN_Small()
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
    parser.add_argument('--n-val', type=int, default=6400, metavar='NV',
                        help='number of validation examples to use')
    parser.add_argument('--batch-size', type=int, default=128, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
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
