#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.ndimage.filters
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from itertools import cycle

import env
from utils import (KaggleCameraDataset, LMDB_Dataset, DatasetIndexer, make_numpy_dataset,
                   RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot)
from utils.pytorch_samplers import StratifiedSampler
from optimizers import ClassificationOptimizer


class CNN2(nn.Module):
    def __init__(self, num_classes=10, input_size=256):
        super(CNN2, self).__init__()

        self.input_size = input_size
        assert self.input_size in [128, 256]

        features = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if self.input_size == 256:
            features += [
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1),
                nn.BatchNorm2d(num_features=256),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        elif self.input_size == 128:
            features += [
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
                nn.BatchNorm2d(num_features=256),
                nn.PReLU()
            ]
        self.features = nn.Sequential(*features)

        n_units = [4096, 256] if self.input_size == 256 else [1024, 128]
        self.classifier = nn.Sequential(
            nn.Linear(n_units[0], n_units[1]),
            nn.PReLU(),
            nn.Linear(n_units[1], num_classes),
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


class CNN3(nn.Module):
    def __init__(self, num_classes=10, input_size=256):
        super(CNN3, self).__init__()

        self.input_size = input_size
        assert self.input_size in [128, 256]

        features = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=7, stride=1),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        ]
        if self.input_size == 256:
            features += [
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
                nn.BatchNorm2d(num_features=1024),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
        self.features = nn.Sequential(*features)

        n_units = [1024, 256] if self.input_size == 256 else [512, 128]
        self.classifier = nn.Sequential(
            nn.Linear(n_units[0], n_units[1]),
            nn.ReLU(inplace=True),
            nn.Linear(n_units[1], num_classes),
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


K = 1/12. * np.array([[-1,  2,  -2,  2, -1],
                      [ 2, -6,   8, -6,  2],
                      [-2,  8, -12,  8, -2],
                      [ 2, -6,   8, -6,  2],
                      [-1,  2,  -2,  2, -1]])

def conv_K(x):
    """
    Parameters
    ----------
    x : (N, N, 3) np.uint8 [0, 255] np.ndarray

    Returns
    -------
    y : (N, N, 3) np.float32 [0.0, 1.0] np.ndarray
    """
    x = x.astype(np.float32) / 255.
    y = np.zeros_like(x)
    y[:, :, 0] = scipy.ndimage.filters.convolve(x[:, :, 0], K)
    y[:, :, 1] = scipy.ndimage.filters.convolve(x[:, :, 1], K)
    y[:, :, 2] = scipy.ndimage.filters.convolve(x[:, :, 2], K)
    return 4. * y


def make_train_loaders(means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5), **kwargs):
    means = list(means)
    stds = list(stds)

    print 'Loading data ...'
    y_train = np.load(os.path.join(kwargs['data_path'], 'y_train.npy'))
    additional_train_ind = np.load(os.path.join(kwargs['data_path'], 'additional_train_ind.npy'))
    additional_val_ind = np.load(os.path.join(kwargs['data_path'], 'additional_val_ind.npy'))
    assert 2750 + len(additional_train_ind) + len(additional_val_ind) == len(y_train)

    # split into train, val
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    y_orig = y_train[:2750]
    train_ind, val_ind = list(skf.split(np.zeros_like(y_orig), y_orig))[kwargs['fold']]

    rng = RNG()
    # noinspection PyTypeChecker
    train_transform = transforms.Compose([
        transforms.RandomCrop(kwargs['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.choice([0.8, 1.0, 1.2]))),
        transforms.Lambda(lambda img: jpg_compress(img, quality=rng.choice([70, 90, 100]))),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(kwargs['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    dataset = LMDB_Dataset(X_path=os.path.join(kwargs['data_path'], 'train.lmdb'),
                           y=y_train)

    bad_indexes = [2976, 5491, 5515, 5595, 5693, 7271, 7449, 7742,
                   13176, 13380, 13559, 13651, 13743, 14207, 14342, 14550]
    additional_train_ind = np.asarray([a for a in additional_train_ind if not a in bad_indexes])
    additional_val_ind = np.asarray([a for a in additional_val_ind if not a in bad_indexes])
    train_ind = np.concatenate((train_ind, additional_train_ind))
    val_ind = np.concatenate((val_ind, additional_val_ind))

    train_dataset = DatasetIndexer(dataset=dataset,
                                   ind=train_ind,
                                   transform=train_transform)
    val_dataset = DatasetIndexer(dataset=dataset,
                                 ind=val_ind,
                                 transform=val_transform)

    # define loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=kwargs['batch_size'],
                              shuffle=False,
                              num_workers=kwargs['n_workers'],
                              sampler=StratifiedSampler(class_vector=y_train[train_ind],
                                                        batch_size=kwargs['batch_size']))
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=kwargs['n_workers'])
    return train_loader, val_loader

def make_train_loaders2(means, stds, folds, **kwargs):
    # assemble data
    y_train = []
    X_train = []
    for fold_id in folds:
        y_train += np.load(os.path.join(kwargs['data_path'], 'y_{0}.npy'.format(fold_id))).tolist()
        X_fold   = np.load(os.path.join(kwargs['data_path'], 'X_{0}.npy'.format(fold_id)))
        X_train += [X_fold[i] for i in xrange(len(X_fold))]
    X_pseudo = np.load(os.path.join(kwargs['data_path'], 'X_pseudo_train.npy'))
    ind = [5*fold_id + i for i in xrange(5) for fold_id in folds]
    X_train += [X_pseudo[i] for i in ind]

    # make dataset
    rng = RNG()
    # noinspection PyTypeChecker
    train_transforms_list = [
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.RandomCrop(kwargs['crop_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.choice([0.8, 1.0, 1.2]))),
        transforms.Lambda(lambda img: jpg_compress(img, quality=rng.choice([70, 90, 100]))),
    ]
    if kwargs['kernel']:
        train_transforms_list += [
            transforms.Lambda(lambda img: conv_K(np.asarray(img, dtype=np.uint8))),
            transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1)))
        ]
    else:
        train_transforms_list += [
            transforms.ToTensor(),
        ]
    train_transforms_list += [transforms.Normalize(means, stds)]
    train_transform = transforms.Compose(train_transforms_list)
    dataset = make_numpy_dataset(X=X_train, y=y_train,
                                 transform=train_transform)

    # make loader
    loader = DataLoader(dataset=dataset,
                        batch_size=kwargs['batch_size'],
                        shuffle=False,
                        num_workers=kwargs['n_workers'],
                        sampler=StratifiedSampler(class_vector=np.asarray(y_train),
                                                  batch_size=kwargs['batch_size']))
    return loader

def train(optimizer, **kwargs):
    # train_loader, val_loader = make_train_loaders(means=(0.5, 0.5, 0.5),
    #                                                stds=(0.5, 0.5, 0.5), **kwargs)
    #
    # print 'Starting training ...'
    # optimizer.train(train_loader, val_loader)
    train2(optimizer, **kwargs)

def train_optimizer(optimizer, train_loader, val_loader, **kwargs):
    optimizer.train(train_loader, val_loader)

def train2(optimizer, means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5),
           train_optimizer=train_optimizer, **kwargs):
    # load and crop validation data
    print "Loading data ..."
    X_val = np.load(os.path.join(kwargs['data_path'], 'X_val.npy'))
    y_val = np.load(os.path.join(kwargs['data_path'], 'y_val.npy')).tolist()
    c = kwargs['crop_size']
    C = X_val.shape[1]
    if c < C:
        X_val = X_val[:, C/2-c/2:C/2+c/2, C/2-c/2:C/2+c/2, :]
    X_val = [X_val[i] for i in xrange(len(X_val))]
    if kwargs['kernel']:
        X_val = [conv_K(x) for x in X_val]

    # compute folds numbers
    fold = kwargs['fold']
    N_folds = 100
    val_folds = [2*fold, 2*fold + 1]
    val_folds.append('pseudo_val')
    train_folds = range(N_folds)[:2*fold] + range(N_folds)[2*fold + 2:]
    G = cycle(train_folds)
    for _ in xrange(kwargs['skip_train_folds']):
        next(G)

    # load val data
    for fold_id in val_folds:
        X_fold = np.load(os.path.join(kwargs['data_path'], 'X_{0}.npy'.format(fold_id)))
        D = X_fold.shape[1]
        X_fold = X_fold[:, D/2-c/2:D/2+c/2, D/2-c/2:D/2+c/2, :]
        Z = [X_fold[i] for i in xrange(len(X_fold))]
        if kwargs['kernel']:
            Z = [conv_K(x) for x in Z]
        X_val += Z
        y_fold = np.load(os.path.join(kwargs['data_path'], 'y_{0}.npy'.format(fold_id))).tolist()
        y_val += y_fold

    # make validation loader
    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: Image.fromarray(x)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: torch.from_numpy(x.transpose(2, 0, 1))),
        transforms.Normalize(means, stds)
    ])
    val_dataset = make_numpy_dataset(X=X_val,
                                     y=y_val,
                                     transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=kwargs['n_workers'])

    n_runs = kwargs['epochs'] / kwargs['epochs_per_unique_data'] + 1

    print "Starting training ..."
    for _ in xrange(n_runs):
        current_folds = []
        for j in xrange(kwargs['n_train_folds']):
            current_folds.append(next(G))

        train_loader = \
            make_train_loaders2(means=means, stds=stds, folds=current_folds, **kwargs)

        optimizer.max_epoch = optimizer.epoch + kwargs['epochs_per_unique_data']
        train_optimizer(optimizer, train_loader, val_loader, **kwargs)

def optical_crop(img, x1, y1, crop_size):
    """
    Depending on the position of the crop,
    rotate it so the the optical center of the camera is in bottom left:
    +--------+
    |        |
    | *      |
    | **     |
    | ***    |
    +--------+
    """
    w = img.size[0]
    h = img.size[1]
    img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    if x1 + crop_size/2 < w/2: # center of crop is the left half
        if y1 + crop_size/2 < h/2: # top-left
            img = img.transpose(Image.ROTATE_270)
        else: # bottom-left
            img = img.transpose(Image.ROTATE_180)
    else: # center of crop is the right half
        if y1 + crop_size / 2 < h / 2:  # top-right
            pass
        else:  # bottom-right
            img = img.transpose(Image.ROTATE_90)
    return img

def random_optical_crop(img, rng, crop_size):
    return optical_crop(img,
                        x1=rng.randint(img.size[0] - crop_size),
                        y1=rng.randint(img.size[1] - crop_size),
                        crop_size=crop_size)

def make_test_dataset_loader(means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5), **kwargs):
    means = list(means)
    stds = list(stds)

    test_transform = transforms.Compose([
        transforms.CenterCrop(kwargs['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    # TTA
    rng = RNG()
    base_transform = transforms.Compose([
        transforms.Lambda(lambda img: random_optical_crop(img, rng, kwargs['crop_size'])),
        # transforms.RandomCrop(kwargs['crop_size']),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.Lambda(lambda img: [img,
        #                                img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.choice([0.8, 1.0, 1.2]))),
        # transforms.Lambda(lambda img: jpg_compress(img, quality=rng.choice([70, 90, 100]))),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    tta_n = kwargs['tta_n']
    def tta_f(img, n=tta_n - 1):
        out = [test_transform(img)]
        for _ in xrange(n):
            out.append(base_transform(img))
        return torch.stack(out, 0)

    tta_transform = transforms.Compose([
        transforms.Lambda(lambda img: tta_f(img)),
    ])

    test_dataset = KaggleCameraDataset(kwargs['data_path'], train=False,
                                       transform=tta_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=kwargs['batch_size'],
                             shuffle=False,
                             num_workers=kwargs['n_workers'])
    return test_dataset, test_loader


def predict(optimizer, means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5), **kwargs):
    test_dataset, test_loader = make_test_dataset_loader(means=means,
                                                         stds=stds,
                                                         **kwargs)

    # compute predictions
    logits, _ = optimizer.test(test_loader)

    # compute and save raw probs
    logits = np.vstack(logits)

    # group and average logits (geom average predictions)
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
    tta_n = kwargs['tta_n']
    logits = logits.reshape(len(logits) / tta_n, tta_n, -1).mean(axis=1)

    proba = softmax(logits)
    # proba = proba.reshape(len(proba)/tta_n, tta_n, -1).mean(axis=1)

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
    model = {'cnn2': CNN2,
             'cnn3': CNN3}[kwargs['model']](input_size=kwargs['crop_size'])

    model_params = [
        {'params': model.features.parameters()},
        {'params': model.classifier.parameters(), 'weight_decay': 1e-5},
    ]
    path_template = os.path.join(kwargs['model_dirpath'], '{acc:.4f}-{epoch}')
    optim = {'adam': torch.optim.Adam,
             'sgd':  torch.optim.SGD}[kwargs['optim']]
    optim_params = dict(lr=kwargs['lr'])
    if kwargs['optim'] == 'sgd':
        optim_params['momentum'] = 0.9
    optimizer = ClassificationOptimizer(model=model, model_params=model_params,
                                        optim=optim, optim_params=optim_params,
                                        loss_func={'logloss': nn.CrossEntropyLoss,
                                                   'hinge': nn.MultiMarginLoss}[kwargs['loss']](),
                                        max_epoch=0, val_each_epoch=kwargs['epochs_per_unique_data'],
                                        cyclic_lr=kwargs['cyclic_lr'],
                                        path_template=path_template)

    if kwargs['predict_from']:
        optimizer.load(kwargs['predict_from'])
        predict(optimizer, **kwargs)
        return

    if kwargs['resume_from']:
        if not kwargs['resume_from'].endswith('ckpt') and not kwargs['resume_from'].endswith('/'):
            kwargs['resume_from'] += '/'
        print 'Resuming from checkpoint ...'
        optimizer.load(kwargs['resume_from'])
        optimizer.path_template = os.path.join(*(list(os.path.split(kwargs['resume_from'])[:-1]) + ['{acc:.4f}-{epoch}']))
        optimizer._mul_lr_by(kwargs['lrm'])

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
    parser.add_argument('--model', type=str, default='cnn2', metavar='PATH',
                        help="model to fine-tune, {'cnn2', 'cnn3'}")
    parser.add_argument('--loss', type=str, default='logloss', metavar='PATH',
                        help="loss function, {'logloss', 'hinge'}")
    parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--optim', type=str, default='adam', metavar='ALGO',
                        help="optimizer, {'adam', 'sgd'}")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate(s)')
    parser.add_argument('--cyclic-lr', type=float, default=None, metavar='CLR', nargs='+',
                        help='cyclic LR in form (lr-min, lr-max, stepsize)')
    parser.add_argument('--epochs', type=int, default=300, metavar='E',
                        help='number of epochs')
    parser.add_argument('--epochs-per-unique-data', type=int, default=8, metavar='EU',
                        help='number of epochs run per unique subset of data')
    parser.add_argument('--lrm', type=float, default=1., metavar='M',
                        help='learning rates multiplier, used only when resume training')
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