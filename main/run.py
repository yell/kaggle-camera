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
parser.add_argument('-cp', '--crop-policy', type=str, default='random',
                    help='crop policy to use for training, {center, random, optical}')
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
parser.add_argument('-bt', '--bootstrap', action='store_true',
                    help='whether to sample from data with replacement (uniformly for each class)')

parser.add_argument('-m', '--model', type=str, default='densenet121',
                    help='model to use')
parser.add_argument('-l', '--loss', type=str, default='logloss',
                    help="loss function, {'logloss', 'hinge'}")
parser.add_argument('-opt', '--optim', type=str, default='sgd',
                    help="optimizer, {'adam', 'sgd'}")
parser.add_argument('-b', '--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('-fc', '--fc-units', type=int, default=(512, 128), nargs='+',
                    help='number of units in FC layers')
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
                    help='checkpoint path to make test predictions from')
parser.add_argument('-pt', '--predict-train', type=str, default=None,
                    help='checkpoint path to make train predictions from')
parser.add_argument('-pv', '--predict-val', type=str, default=None,
                    help='checkpoint path to make val predictions from')


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


N_BLOCKS = [21, 14, 16, 16, 12, 18, 31, 16, 18, 22]
N_IMAGES_PER_CLASS = [991, 651, 767, 773, 595, 873, 1490, 751, 888, 1068]
N_IMAGES_PER_BLOCK = [
    [48, 48, 48, 48, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47],
    [47, 47, 47, 47, 47, 47, 47, 46, 46, 46, 46, 46, 46, 46],
    [48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47],
    [49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48],
    [50, 50, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49],
    [49, 49, 49, 49, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48],
    [49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48],
    [47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 46],
    [50, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49],
    [49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]
]
TRAIN_MANIP_RATIO = 0.5
VAL_MANIP_RATIO = 0.3

# N_BLOCKS = [21, 16, 16, 17, 12, 19, 31, 16, 31, 23]
# N_PSEUDO_BLOCKS = [28, 10, 27, 27, 26, 28, 28, 23, 25, 26]
# N_IMAGES_PER_CLASS = [1014, 746, 767, 807, 598, 918, 1492, 790, 1478, 1081]
# for i in xrange(10):
#     N_IMAGES_PER_CLASS[i] += 24  # images from former validation set
# PSEUDO_IMAGES_PER_CLASS = [224, 79, 213, 218, 212, 228, 227, 182, 199, 205]
# for i in xrange(10):
#     N_IMAGES_PER_CLASS[i] += PSEUDO_IMAGES_PER_CLASS[i]
# N_IMAGES_PER_BLOCK = [
#     [51, 51, 51, 50, 50, 50, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49],
#     [49, 49, 49, 49, 49, 49, 49, 49, 48, 48, 47, 47, 47, 47, 47, 47],
#     [50, 50, 50, 50, 50, 50, 50, 50, 49, 49, 49, 49, 49, 49, 49, 48],
#     [50, 50, 50, 50, 50, 50, 50, 49, 48, 48, 48, 48, 48, 48, 48, 48, 48],
#     [52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 51, 51],
#     [51, 51, 51, 51, 51, 50, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49],
#     [50, 50, 50, 50, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 48, 48, 48, 48, 48, 48, 48],
#     [52, 52, 52, 52, 52, 52, 51, 51, 50, 50, 50, 50, 50, 50, 50, 50],
#     [49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 48, 48, 48, 47, 47, 47, 47, 47, 47, 47],
#     [49, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48]
# ]
# N_IMAGES_PER_PSEUDO_BLOCK = [
#     [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     [8, 8, 8, 8, 8, 8, 8, 8, 8, 7],
#     [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7],
#     [9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     [9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     [9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     [9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7],
#     [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7],
#     [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7]
# ]
b_ind = []
# b_pseudo_ind = []
if args.bootstrap:
    for c in xrange(10):
        b_ind.append([])
        for b in xrange(N_BLOCKS[c]):
            N = N_IMAGES_PER_BLOCK[c][b]
            seed = 42 * args.random_seed + 101 * c + b if args.random_seed else None
            b_ind[c] += [RNG(seed).choice(range(N), N).tolist()]
        # b_pseudo_ind.append([])
        # for b in xrange(N_PSEUDO_BLOCKS[c]):
        #     N = N_IMAGES_PER_PSEUDO_BLOCK[c][b]
        #     seed = 42 * args.random_seed + 1111 * c + b + 1337 if args.random_seed else None
        #     b_pseudo_ind[c] += [RNG(seed).choice(range(N), N).tolist()]


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

def interp(img, ratio='0.5', rng=None, crop_policy=args.crop_policy, crop_size=args.crop_size):
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
        x = make_crop(img, 2 * crop_size, rng, crop_policy=crop_policy)
    elif ratio == '0.8':
        x = make_crop(img, int(crop_size * 1.25 + 1), rng, crop_policy=crop_policy)
    elif ratio == '1.5':
        x = make_crop(img, int(crop_size * 2 / 3 + 1), rng, crop_policy=crop_policy)
    elif ratio == '2.0':
        x = make_crop(img, crop_size / 2, rng, crop_policy=crop_policy)
    else:
        raise ValueError("invalid `ratio`, '{0}'".format(ratio))
    return x.resize((crop_size, crop_size), Image.BICUBIC)

def make_random_manipulation(img, rng, crop_policy=args.crop_policy, crop_size=args.crop_size):
    """
    Parameters
    ----------
    img : 1024x1024 PIL image

    Returns
    -------
    img_manip : (args.crop_size, args.crop_size) PIL image
    """
    return rng.choice([
        lambda x: jpg_compress(make_crop(x, crop_size, rng, crop_policy=crop_policy), quality=70),
        lambda x: jpg_compress(make_crop(x, crop_size, rng, crop_policy=crop_policy), quality=90),
        lambda x: adjust_gamma(make_crop(x, crop_size, rng, crop_policy=crop_policy), gamma=0.8),
        lambda x: adjust_gamma(make_crop(x, crop_size, rng, crop_policy=crop_policy), gamma=1.2),
        lambda x: interp(x, ratio='0.5', rng=rng, crop_policy=crop_policy, crop_size=crop_size),
        lambda x: interp(x, ratio='0.8', rng=rng, crop_policy=crop_policy, crop_size=crop_size),
        lambda x: interp(x, ratio='1.5', rng=rng, crop_policy=crop_policy, crop_size=crop_size),
        lambda x: interp(x, ratio='2.0', rng=rng, crop_policy=crop_policy, crop_size=crop_size),
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
    manip_train = []

    for c in xrange(10):
        X_block = np.load(os.path.join(args.data_path, 'X_{0}_{1}.npy'.format(c, block_index % N_BLOCKS[c])))
        X_block = [X_block[i] for i in xrange(len(X_block))]
        if args.bootstrap:
            X_block = [X_block[i] for i in b_ind[c][block_index % N_BLOCKS[c]]]
        X_train += X_block
        y_train += np.repeat(c, len(X_block)).tolist()
        manip_train += [float32(0.)] * len(X_block)

    # for c in xrange(10):
    #     X_pseudo_block = np.load(os.path.join(args.data_path, 'X_pseudo_{0}_{1}.npy'.format(c, block_index % N_PSEUDO_BLOCKS[c])))
    #     X_pseudo_block = [X_pseudo_block[i] for i in xrange(len(X_pseudo_block))]
    #     if args.bootstrap:
    #         X_pseudo_block = [X_pseudo_block[i] for i in b_pseudo_ind[c][block_index % N_PSEUDO_BLOCKS[c]]]
    #     X_train += X_pseudo_block
    #     y_train += np.repeat(c, len(X_pseudo_block)).tolist()
    #     manip_block = np.load(os.path.join(args.data_path, 'manip_pseudo_{0}_{1}.npy'.format(c, block_index % N_PSEUDO_BLOCKS[c])))
    #     manip_block = [m for m in manip_block]
    #     if args.bootstrap:
    #         manip_block = [manip_block[i] for i in b_pseudo_ind[c][block_index % N_PSEUDO_BLOCKS[c]]]
    #     manip_train += manip_block

    shuffle_ind = range(len(y_train))
    RNG(seed=block_index).shuffle(shuffle_ind)
    X_train = [X_train[i] for i in shuffle_ind]
    y_train = [y_train[i] for i in shuffle_ind]
    manip_train = [manip_train[i] for i in shuffle_ind]

    # make dataset
    rng = RNG(args.random_seed)
    train_transforms_list = [
        transforms.Lambda(lambda (x, m, y): (Image.fromarray(x), m, y)),
        ######
        # 972/1982 manip pseudo images
        # images : pseudo = approx. 48 : 8 = 6 : 1
        # thus to get 50 : 50 manip : unalt we manip 11965/25874 ~ 46% of non-pseudo images
        ######
        transforms.Lambda(lambda (img, m, y): (make_random_manipulation(img, rng), float32(1.), y) if \
                          m[0] < 0.5 and rng.rand() < TRAIN_MANIP_RATIO else (make_crop(img, args.crop_size, rng), m, y)),
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
    dataset = make_numpy_dataset(X=[(x, m, y) for x, m, y in zip(X_train, manip_train, y_train)],
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
    manip_val = np.zeros((len(y_val), 1), dtype=np.float32) # np.load(os.path.join(args.data_path, 'manip_with_pseudo.npy'))  # 68/480 manipulated
    c = args.crop_size
    C = X_val.shape[1]
    if c < C:
        X_val = X_val[:, C/2-c/2:C/2+c/2, C/2-c/2:C/2+c/2, :]
    if args.kernel:
        X_val = [conv_K(x) for x in X_val]

    # make validation loader
    rng = RNG(args.random_seed + 42 if args.random_seed else None)
    val_transform = transforms.Compose([
        transforms.Lambda(lambda (x, m, y): (Image.fromarray(x), m, y)),
        ########
        # 1 - (480-68-0.3*480)/(480-68) ~ 0.18
        ########
        transforms.Lambda(lambda (img, m, y): (make_random_manipulation(img, rng, crop_policy='center'), float32(1.), y) if\
                                               m[0] < 0.5 and rng.rand() < VAL_MANIP_RATIO else (img, m, y)),
        transforms.Lambda(lambda (img, m, y): ([img,
                                                img.transpose(Image.ROTATE_90)][int(rng.rand() < 0.5)], m) if \
                                                KaggleCameraDataset.is_rotation_allowed()[y] else (img, m)),
        transforms.Lambda(lambda (img, m): (transforms.ToTensor()(img), m)),
        transforms.Lambda(lambda (img, m): (transforms.Normalize(args.means, args.stds)(img), m))
    ])
    np.save(os.path.join(args.model_dirpath, 'y_val.npy'), np.vstack(y_val))
    val_dataset = make_numpy_dataset(X=[(x, m, y) for x, m, y in zip(X_val, manip_val, y_val)],
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

def make_test_loader():
    # TTA
    rng = RNG(args.random_seed)
    test_transforms_list = make_aug_transforms(rng, propagate_manip=False)
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
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(args.means, args.stds)(transforms.ToTensor()(crop)) for crop in crops]))
        ]
    test_transform = transforms.Compose(test_transforms_list)
    test_dataset = KaggleCameraDataset(args.data_path, train=False,
                                       transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.n_workers)
    return test_dataset, test_loader

def predict(optimizer):
    test_dataset, test_loader = make_test_loader()

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

def _make_predict_train_loader(X_b, manip_b, manip_ratio=VAL_MANIP_RATIO):
    assert len(X_b) == len(manip_b)

    # make dataset
    rng = RNG(1337)
    train_transforms_list = [
        transforms.Lambda(lambda (x, m): (Image.fromarray(x), m)),
        # if `val` == False
        #   972/1982 manip pseudo images
        #   images : pseudo = approx. 48 : 8 = 6 : 1
        #   to get unalt : manip = 70 : 30 (like in test metric),
        #   we manip ~24.7% of non-pseudo images
        # else:
        #   we simply use same ratio as in validation (0.18)
        transforms.Lambda(lambda (img, m): (make_random_manipulation(img, rng, crop_policy='center', crop_size=512), float32(1.)) if \
                          m[0] < 0.5 and rng.rand() < manip_ratio else (center_crop(img, 512), m))
    ]
    train_transforms_list += make_aug_transforms(rng)
    if args.crop_size == 512:
        train_transforms_list += [
            transforms.Lambda(lambda (img, m): ([img,
                                                 img.transpose(Image.ROTATE_90)], [m] * 2)),
            transforms.Lambda(lambda (crops, ms): (torch.stack(
                [transforms.Normalize(args.means, args.stds)(transforms.ToTensor()(crop)) for crop in crops]), torch.from_numpy(np.asarray(ms))))
        ]
    else:
        train_transforms_list += [
            transforms.Lambda(lambda (img, m): (transforms.TenCrop(args.crop_size)(img), [m] * 10)),
            transforms.Lambda(lambda (imgs, ms): (list(imgs) +
                                                 [img.transpose(Image.ROTATE_90) for img in imgs], ms + ms)),
            transforms.Lambda(lambda (crops, ms): (torch.stack(
                [transforms.Normalize(args.means, args.stds)(transforms.ToTensor()(crop)) for crop in crops]), torch.from_numpy(np.asarray(ms))))
        ]
    train_transform = transforms.Compose(train_transforms_list)
    dataset = make_numpy_dataset(X=[(x, m) for x, m in zip(X_b, manip_b)],
                                 y=np.zeros(len(X_b), dtype=np.int64),
                                 transform=train_transform)

    # make loader
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.n_workers)
    return loader

def _gen_predict_val_loader():
    X_val = np.load(os.path.join(args.data_path, 'X_val_with_pseudo.npy'))
    y_val = np.load(os.path.join(args.data_path, 'y_val_with_pseudo.npy'))
    manip_val = np.load(os.path.join(args.data_path, 'manip_with_pseudo.npy'))
    loader = _make_predict_train_loader(X_val, manip_val, manip_ratio=VAL_MANIP_RATIO)
    yield loader, y_val.tolist(), manip_val

def _gen_predict_train_loaders(max_len=500):
    X_b = []
    y_b = []
    manip_b = []

    for c in xrange(10):
        for b in xrange(N_BLOCKS[c]):
            X_block = np.load(os.path.join(args.data_path, 'X_{0}_{1}.npy'.format(c, b % N_BLOCKS[c])))
            X_b += [X_block[i] for i in xrange(len(X_block))]
            y_b += np.repeat(c, len(X_block)).tolist()
            manip_b += [float32(0.)] * len(X_block)
            if len(y_b) >= max_len:
                yield _make_predict_train_loader(X_b, manip_b), y_b, manip_b
                X_b = []
                y_b = []
                manip_b = []

    # for c in xrange(10):
    #     for b in xrange(N_PSEUDO_BLOCKS[c]):
    #         X_pseudo_block = np.load(os.path.join(args.data_path, 'X_pseudo_{0}_{1}.npy'.format(c, b % N_PSEUDO_BLOCKS[c])))
    #         X_b += [X_pseudo_block[i] for i in xrange(len(X_pseudo_block))]
    #         y_b += np.repeat(c, len(X_pseudo_block)).tolist()
    #         manip_block = np.load(os.path.join(args.data_path, 'manip_pseudo_{0}_{1}.npy'.format(c, b % N_PSEUDO_BLOCKS[c])))
    #         manip_b += [m for m in manip_block]
    #         if len(y_b) >= max_len:
    #             yield _make_predict_train_loader(X_b, manip_b), y_b, manip_b
    #             X_b = []
    #             y_b = []
    #             manip_b = []

    if y_b > 0:
        yield _make_predict_train_loader(X_b, manip_b), y_b, manip_b

def predict_train_val(optimizer, path, val=True):
    logits = []
    y = []
    manip = []
    weights = [2., 1.] if args.crop_size == 512 else [2.] * 10 + [1.] * 10

    block = 0
    for loader_b, y_b, manip_b in (_gen_predict_val_loader() if val else _gen_predict_train_loaders()):
        block += 1
        print "block {0}".format(block)
        logits_b, _ = optimizer.test(loader_b)
        logits_b = np.vstack(logits_b)
        tta_n = len(logits_b) / len(y_b)
        logits_b = logits_b.reshape(len(logits_b) / tta_n, tta_n, -1)
        logits_b = np.average(logits_b, axis=1, weights=weights)
        logits.append(logits_b)
        y += y_b
        manip.append(manip_b)

    logits = np.vstack(logits)
    y = np.asarray(y)
    manip = np.vstack(manip)
    assert len(logits) == len(y) == len(manip)

    dirpath = os.path.split(path)[0]
    suffix = '_val' if val else '_train'
    np.save(os.path.join(dirpath, 'logits{0}.npy'.format(suffix)), logits)
    np.save(os.path.join(dirpath, 'y{0}.npy'.format(suffix)), y)
    np.save(os.path.join(dirpath, 'manip{0}.npy'.format(suffix)), manip)


def main():
    # build model
    if not args.model_dirpath.endswith('/'):
        args.model_dirpath += '/'

    print 'Building model ...'
    model_cls, is_pretrained = get_model(args.model)
    model = model_cls(input_size=args.crop_size, dropout=args.dropout, fc_units=args.fc_units)

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

    patience = 10
    patience *= max(N_BLOCKS) # correction taking into account how the net is trained
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
        print 'Predicting on test set from checkpoint ...'
        optimizer.load(args.predict_from)
        predict(optimizer)
        return

    if args.predict_train:
        if not args.predict_train.endswith('ckpt') and not args.predict_train.endswith('/'):
            args.predict_train += '/'
        print 'Predicting on training set from checkpoint ...'
        optimizer.load(args.predict_train)
        predict_train_val(optimizer, args.predict_train, val=False)
        return

    if args.predict_val:
        if not args.predict_val.endswith('ckpt') and not args.predict_val.endswith('/'):
            args.predict_val += '/'
        print 'Predicting on training set from checkpoint ...'
        optimizer.load(args.predict_val)
        predict_train_val(optimizer, args.predict_val, val=True)
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
