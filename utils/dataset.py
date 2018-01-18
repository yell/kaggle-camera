import os
import numpy as np
import torch.utils.data as data
from PIL import Image


class NumpyDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        x = self.X[index].astype(np.float32)
        x /= 255.
        x -= 0.5
        x *= 2.
        return x, self.y[index]

    def __len__(self):
        return len(self.X)


def make_numpy_dataset(X, y):
    return NumpyDataset(X, y)


class KaggleCameraDataset(data.Dataset):
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
                    target = KaggleCameraDataset.target_labels().index(camera)
                    self.y.append(target)
        else:
            path = os.path.join(root, 'test')
            for fname in sorted(os.listdir(path)):
                x = os.path.join(path, fname)
                if not self.lazy:
                    x = self._load_and_transform(x)
                self.X.append(x)
            self.y = [0] * len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if self.lazy:
            x = self._load_and_transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.X)
