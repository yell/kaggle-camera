import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

from rng import RNG
from utils import batch_iter


def load_images(mode=None, path='.', verbose=False):
    if mode == 'train':
        path = os.path.join(path, 'train')
    elif mode == 'test':
        path = os.path.join(path, 'test')
    for root, dirs, files in os.walk(path):
        for f in files:
            if verbose: print f
            im = Image.open(os.path.join(path, f))
            yield im


class SR_Data(data.Dataset):
    def __init__(self, X, Y, transform=torch.from_numpy):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        lr_data = self.X[index]
        hr_target = self.Y[index]
        return self.transform(lr_data), self.transform(hr_target)

    def __len__(self):
        return min(len(self.X), len(self.Y))


class SR_Loader(object):
    def __init__(self, images, lr_transform=None, hr_transform=None,
                 batch_size=8, n_patches_per_image=8):
        self.images = images
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.batch_size = batch_size
        self.n_patches_per_image = n_patches_per_image

    def __iter__(self):
        for I_b in batch_iter(self.images, self.batch_size):
            X, Y = [], []
            B = len(I_b)
            for i in xrange(B * self.n_patches_per_image):
                X.append( self.lr_transform(I_b[i % B]) )
                Y.append( self.hr_transform(I_b[i % B]) )
            loader = data.DataLoader(SR_Data(X, Y, lambda x: x),
                                     batch_size=B * self.n_patches_per_image,
                                     shuffle=False, num_workers=2)
            for X_b, Y_b in loader:
                yield X_b, Y_b

    def __len__(self):
        N = len(self.images)
        return N / self.batch_size + (N % self.batch_size > 0)


def make_train_loader(data_path, inf_aug=False, interpolate=False, n_samples=None,
                      batch_size=128, crop_size=64, upsample=2, n_train=250, n_patches=16,
                      random_seed=1337):
    """
    Parameters
    ----------
    data_path : str
        Path to the data folder.
    inf_aug : bool, optional
        Whether to generate random patches of images on the fly.

    Returns
    -------
    loader : `torch.utils.data.DataLoader` or `SR_Loader`
    """
    if inf_aug:
        path_BSD = os.path.join(data_path, 'BSDS300', 'images')
        path_Yang = os.path.join(data_path, 'Yang')
        # assemble training data (BSD200 + Yang)
        I = list(load_images(mode='train', path=path_BSD)) + list(load_images(path=path_Yang))
        # filter images larger than `crop_size`^2
        I = filter(lambda im: im.size[0] >= crop_size and im.size[1] >= crop_size, I)
        # randomly shuffle images
        RNG(seed=1337).shuffle(I)
        # select training data
        I_train = I[:n_train]
        # assemble loader
        transform_kwargs = dict(
            crop_size=crop_size,
            upsample=upsample,
            interpolate=interpolate,
            to_numpy=False,
            random_seed=random_seed
        )
        loader = SR_Loader(images=I_train,
                           lr_transform=make_train_transform(lr=True,
                                                             **transform_kwargs),
                           hr_transform=make_train_transform(lr=False,
                                                             **transform_kwargs),
                           batch_size=batch_size,
                           n_patches_per_image=n_patches)
        return loader

    else:
        X_path = 'X_train{0}.npy'.format('_interp' if interpolate else '')
        Y_path = 'Y_train.npy'
        X_path = os.path.join(data_path, X_path)
        Y_path = os.path.join(data_path, Y_path)
        X = np.load(X_path)
        Y = np.load(Y_path)
        if n_samples is not None:
            X = X[:n_samples]
            Y = Y[:n_samples]
        loader = torch.utils.data.DataLoader(SR_Data(X, Y), batch_size=batch_size,
                                             shuffle=False, num_workers=2)
        return loader


def make_test_loader(data_path, dataset='val', interpolate=False, n_samples=None, batch_size=1,
                     crop_data=None, crop_target=None):
    """
    Parameters
    ----------
    data_path : str
        Path to the data folder.
    dataset : {'val', 'bsd100', 'set5', 'set14'}, optional
        Dataset to load

    Returns
    -------
    loader : `torch.utils.data.DataLoader`
    """
    X_path = 'X_{0}{1}.pth'.format(dataset, '_interp' if interpolate else '')
    Y_path = 'Y_{0}.pth'.format(dataset)
    X_path = os.path.join(data_path, X_path)
    Y_path = os.path.join(data_path, Y_path)
    X = torch.load(X_path)
    Y = torch.load(Y_path)
    if n_samples is not None:
        X = X[:n_samples]
        Y = Y[:n_samples]
    if crop_data:
        X = map(lambda x: x[:, :crop_data, :crop_data], X)
    if crop_target:
        Y = map(lambda y: y[:, :crop_target, :crop_target], Y)
    loader = torch.utils.data.DataLoader(SR_Data(X, Y), batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    return loader
