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
from itertools import cycle

import env
from models import get_model
from optimizers import ClassificationOptimizer
from utils import (KaggleCameraDataset, make_numpy_dataset,
                   RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot)
from utils.pytorch_samplers import StratifiedSampler


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-path', type=str, default='../data/',
                    help='directory for storing augmented data etc.')
parser.add_argument('--n-workers', type=int, default=4,
                    help='how many threads to use for I/O')
parser.add_argument('--crop-size', type=int, default=256,
                    help='crop size for patches extracted from training images')
parser.add_argument('--fold', type=int, default=0,
                    help='which fold to use for validation (0-49)')
parser.add_argument('--n-train-folds', type=int, default=4,
                    help='number of fold used for training (each is ~400 Mb)')
parser.add_argument('--skip-train-folds', type=int, default=0,
                    help='how many folds/blocks to skip at the beginning of training')

parser.add_argument('--model', type=str, default='densenet121',
                    help='model to use')
parser.add_argument('--loss', type=str, default='logloss',
                    help="loss function, {'logloss', 'hinge'}")
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--optim', type=str, default='sgd',
                    help="optimizer, {'adam', 'sgd'}")
parser.add_argument('--lr', type=float, default=[1e-4, 1e-3], nargs='+',
                    help='initial learning rate(s)')
parser.add_argument('--cyclic-lr', type=float, default=None, nargs='+',
                    help='cyclic LR in form (lr-min, lr-max, stepsize)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('--epochs-per-unique-data', type=int, default=8,
                    help='number of epochs run per unique subset of data')
parser.add_argument('--lrm', type=float, default=1., nargs='+',
                    help='learning rates multiplier, used only when resume training')
parser.add_argument('--model-dirpath', type=str, default='models/',
                    help='directory path to save the model and predictions')
parser.add_argument('--means', type=float, default=(0.485, 0.456, 0.406), nargs='+',
                    help='per-channel means to use in preprocessing')
parser.add_argument('--stds', type=float, default=(0.229, 0.224, 0.225), nargs='+',
                    help='per-channel standard deviations to use in preprocessing')

parser.add_argument('--resume-from', type=str, default=None,
                    help='checkpoint path to resume training from')
parser.add_argument('--predict-from', type=str, default=None,
                    help='checkpoint path to make predictions from')
parser.add_argument('--tta-n', type=int, default=32,
                    help='number of crops to generate in TTA per test image')
parser.add_argument('--kernel', action='store_true',
                    help='whether to apply kernel for images prior training')
parser.add_argument('--optical', action='store_true',
                    help='whether rotate crops for preserve optical center')

args = parser.parse_args()
args.means = list(args.means)
args.stds = list(args.stds)
kwargs = dict(**vars(args))


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

def make_train_loaders(folds):
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
        transforms.Lambda(lambda x: Image.fromarray(x))
    ]
    if kwargs['optical']:
        train_transforms_list += [
            transforms.Lambda(lambda img: random_optical_crop(img, rng, kwargs['crop_size'])),
            # transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180) if rng.rand() < 0.5 else img),
        ]
    else:
        train_transforms_list += [
            transforms.RandomCrop(kwargs['crop_size']),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.Lambda(lambda img: [img,
            #                                img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        ]
    train_transforms_list += [
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
    train_transforms_list += [transforms.Normalize(kwargs['means'], kwargs['stds'])]
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

def train_optimizer(optimizer, train_loader, val_loader):
    optimizer.train(train_loader, val_loader)

def train_optimizer_pretrained(optimizer, train_loader, val_loader):
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

def train(optimizer, train_optimizer=train_optimizer):
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
        transforms.Normalize(kwargs['means'], kwargs['stds'])
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

        train_loader = make_train_loaders(folds=current_folds)

        optimizer.max_epoch = optimizer.epoch + kwargs['epochs_per_unique_data']
        train_optimizer(optimizer, train_loader, val_loader)

def make_test_dataset_loader():
    test_transform = transforms.Compose([
        transforms.CenterCrop(kwargs['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(kwargs['means'], kwargs['stds'])
    ])

    # TTA
    rng = RNG()
    base_transforms = []
    if kwargs['optical']:
        base_transforms += [
            transforms.Lambda(lambda img: random_optical_crop(img, rng, kwargs['crop_size'])),
            # transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180) if rng.rand() < 0.5 else img),
        ]
    else:
        base_transforms += [
            transforms.RandomCrop(kwargs['crop_size']),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.Lambda(lambda img: [img,
            #                                img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)]),
        ]
    base_transforms += [
        transforms.Lambda(lambda img: adjust_gamma(img, gamma=rng.choice([0.8, 1.0, 1.2]))),
        # transforms.Lambda(lambda img: jpg_compress(img, quality=rng.choice([70, 90, 100]))),
        transforms.ToTensor(),
        transforms.Normalize(kwargs['means'], kwargs['stds'])
    ]
    base_transform = transforms.Compose(*base_transforms)

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

def predict(optimizer):
    test_dataset, test_loader = make_test_dataset_loader()

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

def main():
    # build model
    if not kwargs['model_dirpath'].endswith('/'):
        kwargs['model_dirpath'] += '/'

    print 'Building model ...'
    model, is_pretrained = get_model(kwargs['model'])
    model = model(input_size=kwargs['crop_size'])

    model_params = [
        {'params': model.features.parameters(),
         'lr': kwargs['lr'][0]},
        {'params': model.classifier.parameters(),
         'lr': kwargs['lr'][min(1, len(kwargs['lr']) - 1)],
         'weight_decay': 1e-5},
    ]

    optim = {'adam': torch.optim.Adam,
              'sgd': torch.optim.SGD}[kwargs['optim']]
    optim_params = {'lr': kwargs['lr']}
    if kwargs['optim'] == 'sgd':
        optim_params['momentum'] = 0.9

    path_template = os.path.join(kwargs['model_dirpath'], '{acc:.4f}-{epoch}')
    optimizer = ClassificationOptimizer(model=model, model_params=model_params,
                                        optim=optim, optim_params=optim_params,
                                        loss_func={'logloss': nn.CrossEntropyLoss,
                                                   'hinge': nn.MultiMarginLoss}[kwargs['loss']](),
                                        max_epoch=0, val_each_epoch=kwargs['epochs_per_unique_data'],
                                        cyclic_lr=kwargs['cyclic_lr'],
                                        path_template=path_template)

    if kwargs['predict_from']:
        optimizer.load(kwargs['predict_from'])
        predict(optimizer)
        return

    if kwargs['resume_from']:
        if not kwargs['resume_from'].endswith('ckpt') and not kwargs['resume_from'].endswith('/'):
            kwargs['resume_from'] += '/'
        print 'Resuming from checkpoint ...'
        optimizer.load(kwargs['resume_from'])
        optimizer.path_template = os.path.join(*(list(os.path.split(kwargs['resume_from'])[:-1]) + ['{acc:.4f}-{epoch}']))
        optimizer._mul_lr_by(kwargs['lrm'])

    optimizer.max_epoch = optimizer.epoch + kwargs['epochs']
    train(optimizer,
          train_optimizer=train_optimizer_pretrained if is_pretrained else train_optimizer)


if __name__ == '__main__':
    main()
