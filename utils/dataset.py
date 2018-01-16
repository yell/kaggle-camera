import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

from rng import RNG
from utils import batch_iter


class CameraDataset(data.Dataset):
    @staticmethod
    def target_labels():
        return ['HTC-1-M7',
                'LG-Nexus-5x',
                'Motorola-Droid-Maxx',
                'Motorola-Nexus-6',
                'Motorola-X',
                'Samsung-Galaxy-Note3',
                'Samsung-Galaxy-S4',
                'Sony-NEX-7',
                'iPhone-4s',
                'iPhone-6']

    def __init__(self, root, train=True):
        self.X_files = []
        self.y = []
        if train:
            path = os.path.join(root, 'train')
            for camera in os.listdir(path):
                path_camera = os.path.join(path, camera)
                for fname in sorted(os.listdir(path_camera)):
                    self.X_files.append(os.path.join(path_camera, fname))
                    self.y.append(CameraDataset.target_labels().index(camera))
        else:
            path = os.path.join(root, 'test')
            for fname in sorted(os.listdir(path)):
                self.X_files.append(os.path.join(path, fname))
            self.y = [None] * len(self.X_files)

    def __getitem__(self, index):
        return Image.open(self.X_files[index]), self.y[index]

    def __len__(self):
        return len(self.X_files)


# class SR_Loader(object):
#     def __init__(self, images, lr_transform=None, hr_transform=None,
#                  batch_size=8, n_patches_per_image=8):
#         self.images = images
#         self.lr_transform = lr_transform
#         self.hr_transform = hr_transform
#         self.batch_size = batch_size
#         self.n_patches_per_image = n_patches_per_image
#
#     def __iter__(self):
#         for I_b in batch_iter(self.images, self.batch_size):
#             X, Y = [], []
#             B = len(I_b)
#             for i in xrange(B * self.n_patches_per_image):
#                 X.append( self.lr_transform(I_b[i % B]) )
#                 Y.append( self.hr_transform(I_b[i % B]) )
#             loader = data.DataLoader(SR_Data(X, Y, lambda x: x),
#                                      batch_size=B * self.n_patches_per_image,
#                                      shuffle=False, num_workers=2)
#             for X_b, Y_b in loader:
#                 yield X_b, Y_b
#
#     def __len__(self):
#         N = len(self.images)
#         return N / self.batch_size + (N % self.batch_size > 0)


# def make_train_loader(data_path, inf_aug=False, interpolate=False, n_samples=None,
#                       batch_size=128, crop_size=64, upsample=2, n_train=250, n_patches=16,
#                       random_seed=1337):
#     """
#     Parameters
#     ----------
#     data_path : str
#         Path to the data folder.
#     inf_aug : bool, optional
#         Whether to generate random patches of images on the fly.
#
#     Returns
#     -------
#     loader : `torch.utils.data.DataLoader` or `SR_Loader`
#     """
#     if inf_aug:
#         path_BSD = os.path.join(data_path, 'BSDS300', 'images')
#         path_Yang = os.path.join(data_path, 'Yang')
#         # assemble training data (BSD200 + Yang)
#         I = list(load_images(mode='train', path=path_BSD)) + list(load_images(path=path_Yang))
#         # filter images larger than `crop_size`^2
#         I = filter(lambda im: im.size[0] >= crop_size and im.size[1] >= crop_size, I)
#         # randomly shuffle images
#         RNG(seed=1337).shuffle(I)
#         # select training data
#         I_train = I[:n_train]
#         # assemble loader
#         transform_kwargs = dict(
#             crop_size=crop_size,
#             upsample=upsample,
#             interpolate=interpolate,
#             to_numpy=False,
#             random_seed=random_seed
#         )
#         loader = SR_Loader(images=I_train,
#                            lr_transform=make_train_transform(lr=True,
#                                                              **transform_kwargs),
#                            hr_transform=make_train_transform(lr=False,
#                                                              **transform_kwargs),
#                            batch_size=batch_size,
#                            n_patches_per_image=n_patches)
#         return loader
#
#     else:
#         X_path = 'X_train{0}.npy'.format('_interp' if interpolate else '')
#         Y_path = 'Y_train.npy'
#         X_path = os.path.join(data_path, X_path)
#         Y_path = os.path.join(data_path, Y_path)
#         X = np.load(X_path)
#         Y = np.load(Y_path)
#         if n_samples is not None:
#             X = X[:n_samples]
#             Y = Y[:n_samples]
#         loader = torch.utils.data.DataLoader(SR_Data(X, Y), batch_size=batch_size,
#                                              shuffle=False, num_workers=2)
#         return loader
#
#
# def make_test_loader(data_path, dataset='val', interpolate=False, n_samples=None, batch_size=1,
#                      crop_data=None, crop_target=None):
#     """
#     Parameters
#     ----------
#     data_path : str
#         Path to the data folder.
#     dataset : {'val', 'bsd100', 'set5', 'set14'}, optional
#         Dataset to load
#
#     Returns
#     -------
#     loader : `torch.utils.data.DataLoader`
#     """
#     X_path = 'X_{0}{1}.pth'.format(dataset, '_interp' if interpolate else '')
#     Y_path = 'Y_{0}.pth'.format(dataset)
#     X_path = os.path.join(data_path, X_path)
#     Y_path = os.path.join(data_path, Y_path)
#     X = torch.load(X_path)
#     Y = torch.load(Y_path)
#     if n_samples is not None:
#         X = X[:n_samples]
#         Y = Y[:n_samples]
#     if crop_data:
#         X = map(lambda x: x[:, :crop_data, :crop_data], X)
#     if crop_target:
#         Y = map(lambda y: y[:, :crop_target, :crop_target], Y)
#     loader = torch.utils.data.DataLoader(SR_Data(X, Y), batch_size=batch_size,
#                                          shuffle=False, num_workers=2)
#     return loader
