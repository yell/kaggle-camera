import os
import lmdb
import torch.utils.data as data
from PIL import Image


class LMDB_Dataset(data.Dataset):
    def __init__(self, X_path, y, len=2750, mode='RGB', size=(1024, 1024)):
        self.X_env = lmdb.open(X_path, readonly=True)
        self.y = y
        self.len = len
        self.mode = mode
        self.size = size

    def __getitem__(self, index):
        with self.X_env.begin() as txn:
            bytes = txn.get('{:06}'.format(index))
            x = Image.frombytes('RGB', (1024, 1024), bytes)
            return x, self.y[index]

    def __len__(self):
        return self.len


class NumpyDataset(data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]

    def __len__(self):
        return len(self.X)


def make_numpy_dataset(X, y, transform=None):
    return NumpyDataset(X, y, transform)


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
