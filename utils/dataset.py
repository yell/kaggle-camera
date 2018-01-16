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

    def _load_and_transform(self, x):
        """
        Parameters
        ----------
        x : str
            path to image
        """
        x = Image.open(x)
        if self.transform:
            x = self.transform(x)
        return x

    def __init__(self, root, train=True, lazy=True, transform=None):
        self.X = []
        self.y = []
        self.train = train
        self.lazy = lazy
        self.transform = transform
        if train:
            path = os.path.join(root, 'train')
            for camera in os.listdir(path):
                path_camera = os.path.join(path, camera)
                for fname in sorted(os.listdir(path_camera)):
                    x = os.path.join(path_camera, fname)
                    if not self.lazy:
                        x = self._load_and_transform(x)
                    self.X.append(x)
                    target = CameraDataset.target_labels().index(camera)
                    self.y.append(target)
        else:
            path = os.path.join(root, 'test')
            for fname in sorted(os.listdir(path)):
                self.X.append(os.path.join(path, fname))
            self.y = [None] * len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.lazy:
            x = self._load_and_transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.X)
