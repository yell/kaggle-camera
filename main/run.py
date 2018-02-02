#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.ndimage.filters
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader
from torchvision import transforms

import env
from models import get_model
from optimizers import ClassificationOptimizer, ReduceLROnPlateau
from utils import (KaggleCameraDataset, make_numpy_dataset,
                   RNG, adjust_gamma, jpg_compress,
                   softmax, one_hot_decision_function, unhot, float32)
from utils.pytorch_samplers import StratifiedSampler


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-dd', '--data-path', type=str, default='../data/',
                    help='directory for storing augmented data etc.')
parser.add_argument('-nw', '--n-workers', type=int, default=4,
                    help='how many threads to use for I/O')
# parser.add_argument('-f', '--fold', type=int, default=0,
#                     help='which fold to use for validation (0-49)')
# parser.add_argument('-nb', '--n-blocks', type=int, default=4,
#                     help='number of blocks used for training (each is ~475 Mb)')
# parser.add_argument('-sb', '--skip-blocks', type=int, default=0,
#                     help='how many folds/blocks to skip at the beginning of training')
parser.add_argument('-npc', '--n-img-per-class', type=int, default=None,
                    help='if enabled, how many full JPG images per class to load at once, o/w load blocks')
parser.add_argument('-cp', '--crop-policy', type=str, default='random',
                    help='crop policy to use for training or testing, {center, random, optical}')
parser.add_argument('-ap', '--aug-policy', type=str, default='no-op',
                    help='further augmentation to use for training or testing, {no-op, horiz, d4}')
parser.add_argument('-cs', '--crop-size', type=int, default=256,
                    help='crop size for patches extracted from training images')
parser.add_argument('-k', '--kernel', action='store_true',
                    help='whether to apply kernel for images prior training')
parser.add_argument('--means', type=float, default=(0.485, 0.456, 0.406), nargs='+',
                    help='per-channel means to use in preprocessing')
parser.add_argument('--stds', type=float, default=(0.229, 0.224, 0.225), nargs='+',
                    help='per-channel standard deviations to use in preprocessing')
parser.add_argument('-rs', '--random_seed', type=int, default=None,
                    help='random seed to control data augmentation and manipulations')

parser.add_argument('-m', '--model', type=str, default='densenet121',
                    help='model to use')
parser.add_argument('-l', '--loss', type=str, default='logloss',
                    help="loss function, {'logloss', 'hinge'}")
parser.add_argument('-opt', '--optim', type=str, default='sgd',
                    help="optimizer, {'adam', 'sgd'}")
parser.add_argument('-b', '--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('-d', '--dropout', type=float, default=0.,
                    help='dropout for FC layers')
parser.add_argument('-lr', '--lr', type=float, default=[1e-3], nargs='+',
                    help='initial learning rate(s)')
parser.add_argument('-lrm', '--lrm', type=float, default=[1.], nargs='+',
                    help='learning rates multiplier(s), used only when resume training')
parser.add_argument('-clr', '--cyclic-lr', type=float, default=None, nargs='+',
                    help='cyclic LR in form (lr-min, lr-max, stepsize)')
parser.add_argument('-e', '--epochs', type=int, default=300,
                    help='number of epochs')
parser.add_argument('-eu', '--epochs-per-unique-data', type=int, default=8,
                    help='number of epochs run per unique subset of data')
parser.add_argument('-w', '--weighted', action='store_true',
                    help='whether to use class-weighted loss function')

parser.add_argument('-md', '--model-dirpath', type=str, default='../models/',
                    help='directory path to save the model and predictions')
parser.add_argument('-ct', '--ckpt-template', type=str, default='{acc:.4f}-{epoch}',
                    help='model checkpoint naming template')

parser.add_argument('-rf', '--resume-from', type=str, default=None,
                    help='checkpoint path to resume training from')
parser.add_argument('-pf', '--predict-from', type=str, default=None,
                    help='checkpoint path to make predictions from')


args = parser.parse_args()

args.means = list(args.means)
args.stds = list(args.stds)
if len(args.lr) == 1:
    args.lr *= 2
if len(args.lrm) == 1:
    args.lrm *= 2
args.crop_policy = args.crop_policy.lower()
args.aug_policy = args.aug_policy.lower()
args.model = args.model.lower()
args.loss = args.loss.lower()
args.optim = args.optim.lower()

N_BLOCKS = [21, 16, 16, 17, 12, 19, 31, 16, 31, 23]

with open(os.path.join(args.data_path, 'train_links.json')) as f:
    train_links = json.load(f)
N_IMAGES_PER_CLASS = map(len, train_links.values())
assert N_IMAGES_PER_CLASS == [746, 1014, 807, 767, 918, 598, 790, 1492, 1081, 1478]

for i in xrange(10):
    train_links[i] = map(lambda s: os.path.join(args.data_path, s.replace('../data/', '')), train_links[str(i)])
    del train_links[str(i)]


K = 1/12. * np.array([[-1,  2,  -2,  2, -1],
                      [ 2, -6,   8, -6,  2],
                      [-2,  8, -12,  8, -2],
                      [ 2, -6,   8, -6,  2],
                      [-1,  2,  -2,  2, -1]])

def center_crop(img, crop_size):
    w = img.size[0]
    h = img.size[1]
    return img.crop((w / 2 - crop_size / 2, h / 2 - crop_size / 2,
                     w / 2 + crop_size / 2, h / 2 + crop_size / 2))

def random_crop(img, crop_size, rng):
    x1 = rng.randint(img.size[0] - crop_size) if img.size[0] > crop_size else 0
    y1 = rng.randint(img.size[1] - crop_size) if img.size[1] > crop_size else 0
    return img.crop((x1, y1, x1 + crop_size, y1 + crop_size))

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

def random_optical_crop(img, crop_size, rng):
    return optical_crop(img,
                        x1=rng.randint(img.size[0] - crop_size) if img.size[0] - crop_size > 0 else 0,
                        y1=rng.randint(img.size[1] - crop_size) if img.size[1] - crop_size > 0 else 0,
                        crop_size=crop_size)

def make_crop(img, crop_size, rng, crop_policy=args.crop_policy):
    if crop_policy == 'center':
        return center_crop(img, crop_size)
    if crop_policy == 'random':
        return random_crop(img, crop_size, rng)
    if crop_policy == 'optical':
        return random_optical_crop(img, crop_size, rng)
    raise ValueError("invalid `crop_policy`, '{0}'".format(args.crop_policy))

def interp(img, ratio='0.5', rng=None, crop_policy=args.crop_policy):
    """
    Parameters
    ----------
    img : (1024, 1024) PIL image
    ratio : {'0.5', '0.8', '1.5', '2.0'}

    Returns
    -------
    img_interp : (args.crop_size, args.crop_size) PIL image
    """
    if ratio == '0.5':
        x = make_crop(img, 2 * args.crop_size, rng, crop_policy=crop_policy)
    elif ratio == '0.8':
        x = make_crop(img, int(args.crop_size * 1.25 + 1), rng, crop_policy=crop_policy)
    elif ratio == '1.5':
        x = make_crop(img, int(args.crop_size * 2 / 3 + 1), rng, crop_policy=crop_policy)
    elif ratio == '2.0':
        x = make_crop(img, args.crop_size / 2, rng, crop_policy=crop_policy)
    else:
        raise ValueError("invalid `ratio`, '{0}'".format(ratio))
    return x.resize((args.crop_size, args.crop_size), Image.BICUBIC)

def make_random_manipulation(img, rng, crop_policy=args.crop_policy):
    """
    Parameters
    ----------
    img : 1024x1024 PIL image

    Returns
    -------
    img_manip : (args.crop_size, args.crop_size) PIL image
    """
    return rng.choice([
        # lambda x: jpg_compress(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), quality=70),
        # lambda x: jpg_compress(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), quality=90),
        lambda x: jpg_compress(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), quality=rng.randint(70, 90 + 1)),
        lambda x: jpg_compress(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), quality=rng.randint(70, 90 + 1)),
        # lambda x: adjust_gamma(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), gamma=0.8),
        # lambda x: adjust_gamma(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), gamma=1.2),
        lambda x: adjust_gamma(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), gamma=rng.uniform(0.8, 1.2)),
        lambda x: adjust_gamma(make_crop(x, args.crop_size, rng, crop_policy=crop_policy), gamma=rng.uniform(0.8, 1.2)),
        lambda x: interp(x, ratio='0.5', rng=rng, crop_policy=crop_policy),
        lambda x: interp(x, ratio='0.8', rng=rng, crop_policy=crop_policy),
        lambda x: interp(x, ratio='1.5', rng=rng, crop_policy=crop_policy),
        lambda x: interp(x, ratio='2.0', rng=rng, crop_policy=crop_policy),
    ])(img)

def make_aug_transforms(rng, propagate_manip=True):
    aug_policies = {}
    aug_policies['no-op'] = []
    if propagate_manip:
        aug_policies['horiz'] = [
            transforms.Lambda(lambda (img, m): (img.transpose(Image.FLIP_LEFT_RIGHT) if rng.rand() < 0.5 else img, m))
        ]
    else:
        aug_policies['horiz'] = [
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if rng.rand() < 0.5 else img)
        ]
    if propagate_manip:
        aug_policies['d4'] = [
            transforms.Lambda(lambda (img, m): (img.transpose(Image.FLIP_LEFT_RIGHT) if rng.rand() < 0.5 else img, m)),
            transforms.Lambda(lambda (img, m): (img.transpose(Image.FLIP_TOP_BOTTOM) if rng.rand() < 0.5 else img, m)),
            transforms.Lambda(lambda (img, m): ([img,
                                                 img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)], m))
        ]
    else:
        aug_policies['d4'] = [
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if rng.rand() < 0.5 else img),
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM) if rng.rand() < 0.5 else img),
            transforms.Lambda(lambda img: [img,
                                           img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)])
        ]
    return aug_policies[args.aug_policy]

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
    return y

def make_train_loaders(block_index):
    # assemble data
    X_train = []
    y_train = []
    if args.n_img_per_class:
        ind = np.arange(block_index * args.n_img_per_class,
                        (block_index + 1) * args.n_img_per_class, dtype=np.int32)
        for c in xrange(10):
            for x_link in [train_links[c][l] for l in ind % N_IMAGES_PER_CLASS[c]]:
                img = Image.open(x_link)
                X_train.append(img)
            y_train += np.repeat(c, args.n_img_per_class).tolist()
    else:
        for c in xrange(10):
            X_block = np.load(os.path.join(args.data_path, 'X_{0}_{1}.npy'.format(c, block_index % N_BLOCKS[c])))
            X_train += [X_block[i] for i in xrange(len(X_block))]
            y_train += np.repeat(c, len(X_block)).tolist()

    shuffle_ind = range(len(y_train))
    RNG(seed=block_index).shuffle(shuffle_ind)
    X_train = [X_train[i] for i in shuffle_ind]
    y_train = [y_train[i] for i in shuffle_ind]

    # X_pseudo = np.load(os.path.join(args.data_path, 'X_pseudo_train.npy'))
    # ind = [5*fold_id + i for i in xrange(5) for fold_id in folds]
    # X_train += [X_pseudo[i] for i in ind]

    # make dataset
    rng = RNG(args.random_seed)
    train_transforms_list = []
    if args.n_img_per_class:
        # train_transforms_list.append(transforms.Lambda(lambda (x, y): (x, y)))
        pass
    else:
        train_transforms_list.append(transforms.Lambda(lambda (x, y): (Image.fromarray(x), y)))
    train_transforms_list += [
        transforms.Lambda(lambda (img, y): rng.choice([
            lambda x: (make_crop(x, args.crop_size, rng), float32(0.), y),
            lambda x: (make_random_manipulation(x, rng), float32(1.), y)
        ])(img)),
        transforms.Lambda(lambda (img, m, y): ([img,
                                                img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)], m) if \
                                                KaggleCameraDataset.is_rotation_allowed()[y] else (img, m)),
    ]
    train_transforms_list += make_aug_transforms(rng)

    if args.kernel:
        train_transforms_list += [
            transforms.Lambda(lambda (img, m): (conv_K(np.asarray(img, dtype=np.uint8)), m)),
            transforms.Lambda(lambda (x, m): (torch.from_numpy(x.transpose(2, 0, 1)), m))
        ]
    else:
        train_transforms_list += [
            transforms.Lambda(lambda (img, m): (transforms.ToTensor()(img), m))
        ]
    train_transforms_list += [
        transforms.Lambda(lambda (img, m): (transforms.Normalize(args.means, args.stds)(img), m))
    ]
    train_transform = transforms.Compose(train_transforms_list)
    dataset = make_numpy_dataset(X=[(x, y) for x, y in zip(X_train, y_train)],
                                 y=y_train,
                                 transform=train_transform)

    # make loader
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.n_workers,
                        sampler=StratifiedSampler(class_vector=np.asarray(y_train),
                                                  batch_size=args.batch_size))
    return loader

def train_optimizer(optimizer, train_loader, val_loader):
    optimizer.train(train_loader, val_loader)

def train_optimizer_pretrained(optimizer, train_loader, val_loader):
    if optimizer.epoch == 0:
        # freeze features for the first epoch
        for param in optimizer.optim.param_groups[0]['params']:
            param.requires_grad = False

        max_epoch = optimizer.max_epoch
        optimizer.max_epoch = optimizer.epoch + args.epochs_per_unique_data
        optimizer.train(train_loader, val_loader)

        # now unfreeze features
        for param in optimizer.optim.param_groups[0]['params']:
            param.requires_grad = True

        optimizer.max_epoch = max_epoch
    optimizer.train(train_loader, val_loader)

def train(optimizer, train_optimizer=train_optimizer):
    # load and crop validation data
    print "Loading data ..."
    X_val = np.load(os.path.join(args.data_path, 'X_val.npy'))
    y_val = np.load(os.path.join(args.data_path, 'y_val.npy'))
    c = args.crop_size
    C = X_val.shape[1]
    if c < C:
        X_val = X_val[:, C/2-c/2:C/2+c/2, C/2-c/2:C/2+c/2, :]
    # X_val = [X_val[i] for i in xrange(len(X_val))]
    if args.kernel:
        X_val = [conv_K(x) for x in X_val]

    # # compute folds numbers
    # fold = args.fold
    # N_folds = 100
    # val_folds = [2*fold, 2*fold + 1]
    # # val_folds.append('pseudo_val')
    # train_folds = range(N_folds)[:2*fold] + range(N_folds)[2*fold + 2:]
    # G = cycle(train_folds)
    # for _ in xrange(args.skip_blocks):
    #     next(G)
    #
    # # load val data
    # for fold_id in val_folds:
    #     X_fold = np.load(os.path.join(args.data_path, 'X_{0}.npy'.format(fold_id)))
    #     # D = X_fold.shape[1]
    #     # X_fold = X_fold[:, D/2-c/2:D/2+c/2, D/2-c/2:D/2+c/2, :]
    #     Z = [X_fold[i] for i in xrange(len(X_fold))]
    #     if args.kernel:
    #         Z = [conv_K(x) for x in Z]
    #     X_val += Z
    #     y_fold = np.load(os.path.join(args.data_path, 'y_{0}.npy'.format(fold_id))).tolist()
    #     y_val += y_fold

    # make validation loader
    rng = RNG(args.random_seed + 42 if args.random_seed else None)
    val_transform = transforms.Compose([
        transforms.Lambda(lambda (x, y): (Image.fromarray(x), y)),
        transforms.Lambda(lambda (img, y): (img, float32(0.), y)),
        # transforms.Lambda(lambda (img, y): (center_crop(img, args.crop_size), float32(0.), y)),
        # transforms.Lambda(lambda (img, y): (center_crop(img, args.crop_size), float32(0.), y) if rng.rand() < 0.7 else \
        #                               (make_random_manipulation(img, rng, crop_policy='center'), float32(1.), y)),
        transforms.Lambda(lambda (img, m, y): ([img,
                                                img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)], m) if \
                                                KaggleCameraDataset.is_rotation_allowed()[y] else (img, m)),
        transforms.Lambda(lambda (img, m): (transforms.ToTensor()(img), m)),
        transforms.Lambda(lambda (img, m): (transforms.Normalize(args.means, args.stds)(img), m))
    ])
    np.save(os.path.join(args.model_dirpath, 'y_val.npy'), np.vstack(y_val))
    val_dataset = make_numpy_dataset(X=[(x, y) for x, y in zip(X_val, y_val)],
                                     y=y_val,
                                     transform=val_transform)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_workers)

    n_runs = args.epochs / args.epochs_per_unique_data + 1

    for _ in xrange(n_runs):
        train_loader = make_train_loaders(block_index=optimizer.epoch / args.epochs_per_unique_data)
        optimizer.max_epoch = optimizer.epoch + args.epochs_per_unique_data
        train_optimizer(optimizer, train_loader, val_loader)

def make_test_dataset_loader():
    # TTA
    rng = RNG(args.random_seed)
    # test_transforms_list = [
    #     transforms.Lambda(lambda img: make_crop(img, args.crop_size, rng)),
    #     transforms.Lambda(lambda img: [img,
    #                                    img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)])
    # ]
    test_transforms_list = []
    if args.crop_size == 512:
        test_transforms_list += [
            transforms.Lambda(lambda img: [img,
                                           img.transpose(Image.ROTATE_90)]),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize(args.means, args.stds)(transforms.ToTensor()(crop)) for crop in crops]))
        ]
    else:
        test_transforms_list += [
            transforms.TenCrop(args.crop_size),
            transforms.Lambda(lambda imgs: list(imgs) +\
                                           [img.transpose(Image.ROTATE_90) for img in imgs]),
                                           # [img.transpose(Image.ROTATE_180) for img in imgs] +\
                                           # [img.transpose(Image.ROTATE_270) for img in imgs]),
            # transforms.Lambda(lambda imgs: [[img,
            #                                  img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)] for img in imgs]),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(args.means, args.stds)(transforms.ToTensor()(crop)) for crop in crops]))
        ]

    test_transforms_list += make_aug_transforms(rng, propagate_manip=False)
    # test_transforms_list += [
    #     transforms.ToTensor(),
        # transforms.Normalize(args.means, args.stds)
    # ]
    test_transform = transforms.Compose(test_transforms_list)

    # def tta_f(img, n=args.tta_n):
    #     out = []
    #     for _ in xrange(n):
    #         out.append(test_transform(img))
    #     return torch.stack(out, 0)
    #
    # tta_transform = transforms.Compose([
    #     transforms.Lambda(lambda img: tta_f(img)),
    # ])
    tta_transform = test_transform

    test_dataset = KaggleCameraDataset(args.data_path, train=False,
                                       transform=tta_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.n_workers)
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
    tta_n = len(logits) / 2640
    logits = logits.reshape(len(logits) / tta_n, tta_n, -1)
    weights = [2.,1.] if args.crop_size == 512 else [2.]*10+[1.]*10
    logits = np.average(logits, axis=1, weights=weights)

    proba = softmax(logits)
    # proba = proba.reshape(len(proba)/tta_n, tta_n, -1).mean(axis=1)

    fnames = [os.path.split(fname)[-1] for fname in test_dataset.X]
    df = pd.DataFrame(proba)
    df['fname'] = fnames
    df = df[['fname'] + range(10)]
    dirpath = os.path.split(args.predict_from)[0]
    df.to_csv(os.path.join(dirpath, 'proba.csv'), index=False)

    # compute predictions and save in submission format
    index_pred = unhot(one_hot_decision_function(proba))
    data = {'fname': fnames,
            'camera': [KaggleCameraDataset.target_labels()[int(c)] for c in index_pred]}
    df2 = pd.DataFrame(data, columns=['fname', 'camera'])
    df2.to_csv(os.path.join(dirpath, 'submission.csv'), index=False)

def main():
    # build model
    if not args.model_dirpath.endswith('/'):
        args.model_dirpath += '/'

    print 'Building model ...'
    model_cls, is_pretrained = get_model(args.model)
    model = model_cls(input_size=args.crop_size, dropout=args.dropout)

    model_params = [
        {'params': model.features.parameters(), 'lr': args.lr[0]},
        {'params': model.classifier.parameters(), 'lr': args.lr[1], 'weight_decay': 1e-5},
    ]

    optim = {'adam': torch.optim.Adam,
              'sgd': torch.optim.SGD}[args.optim]
    optim_params = {'lr': args.lr[0]}
    if args.optim == 'sgd':
        optim_params['momentum'] = 0.9

    path_template = os.path.join(args.model_dirpath, args.ckpt_template)

    patience = 5
    # correction taking into account how the net is trained
    if args.n_img_per_class:
        patience *= max(N_IMAGES_PER_CLASS) / args.n_img_per_class
    else:
        patience *= max(N_BLOCKS)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=patience, min_lr=1e-8, eps=1e-6, verbose=1)

    class_weights = np.ones(10)
    if args.weighted:
        class_weights = 1. / np.asarray(N_IMAGES_PER_CLASS)
    class_weights /= class_weights.sum()
    optimizer = ClassificationOptimizer(model=model, model_params=model_params,
                                        optim=optim, optim_params=optim_params,
                                        loss_func={'logloss': nn.CrossEntropyLoss,
                                                   'hinge': nn.MultiMarginLoss}[args.loss],
                                        class_weights=class_weights,
                                        max_epoch=0, val_each_epoch=args.epochs_per_unique_data,
                                        cyclic_lr=args.cyclic_lr, path_template=path_template,
                                        callbacks=[reduce_lr])

    if args.predict_from:
        if not args.predict_from.endswith('ckpt') and not args.predict_from.endswith('/'):
            args.predict_from += '/'
        print 'Predicting from checkpoint ...'
        optimizer.load(args.predict_from)
        predict(optimizer)
        return

    if args.resume_from:
        if not args.resume_from.endswith('ckpt') and not args.resume_from.endswith('/'):
            args.resume_from += '/'
        print 'Resuming from checkpoint ...'
        optimizer.load(args.resume_from)
        optimizer.dirpath = os.path.join(*(list(os.path.split(args.resume_from)[:-1])))
        optimizer.path_template = os.path.join(optimizer.dirpath, args.ckpt_template)
        optimizer._mul_lr_by(args.lrm)
    else:
        print 'Starting training ...'

    optimizer.max_epoch = optimizer.epoch + args.epochs
    train(optimizer,
          train_optimizer=train_optimizer_pretrained if is_pretrained else train_optimizer)


if __name__ == '__main__':
    main()
