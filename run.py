import os
import argparse
import skimage.exposure
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import transforms
from torchvision.models.densenet import densenet121

from utils import CameraDataset, RNG

# TODO: load densenet
# TODO: implement `that` fine-tuning
# TODO: checkpoints handling


def train(**kwargs):
    # load training data
    dataset = CameraDataset(kwargs['data_path'], train=True, lazy=not kwargs['not_lazy'])

    # define train and val transforms
    rng = RNG()
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: [img,
                                       img.transpose(Image.ROTATE_90)][rng.rand() < 0.5]),
        # TODO: transforms.Lambda(lambda img: skimage.exposure.adjust_gamma(img, gamma=0.8-1.2)),
        # TODO: random jpg compression (70-100)
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.ToTensor()
    ])

    # split into train, val in stratified fashion
    sss = StratifiedShuffleSplit(n_splits=1, test_size=kwargs['n_val'],
                                 random_state=kwargs['random_seed'])
    train_index, val_index = list(sss.split(dataset.X, dataset.y))[0]
    train_dataset = CameraDataset(kwargs['data_path'], train=True, lazy=True, transform=train_transform)
    val_dataset   = CameraDataset(kwargs['data_path'], train=True, lazy=True, transform=val_transform)
    train_dataset.X = [dataset.X[i] for i in train_index]
    train_dataset.y = [dataset.y[i] for i in train_index]
    val_dataset.X = [dataset.X[i] for i in val_index]
    val_dataset.y = [dataset.y[i] for i in val_index]

    # define loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=kwargs['batch_size'],
                              shuffle=False,
                              num_workers=4)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=kwargs['batch_size'],
                            shuffle=False,
                            num_workers=4)

    # create model dirpath if needed
    if not os.path.exists(kwargs['model_dirpath']):
        os.makedirs(kwargs['model_dirpath'])

def predict(kwargs):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = CameraDataset(kwargs['data_path'], train=False, lazy=False,
                            transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=kwargs['batch_size'],
                             shuffle=False,
                             num_workers=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-path', type=str, default='data/', metavar='PATH',
                        help='directory for storing augmented data etc.')
    parser.add_argument('--not-lazy', action='store_true',
                        help='if enabled, load all training data into RAM')
    parser.add_argument('--n-val', type=int, default=300, metavar='NV',
                        help='number of validation examples to use')
    parser.add_argument('--lr', type=float, default=[1e-4, 1e-3], metavar='LR', nargs='+',
                        help='initial learning rate(s)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs per unique data')
    parser.add_argument('--lrm', type=float, default=[1., 1.], metavar='M', nargs='+',
                        help='learning rates multiplier(s), used only when resume training')
    parser.add_argument('--random-seed', type=int, default=1337, metavar='N',
                        help='random seed for train-val split')
    parser.add_argument('--model-dirpath', type=str, default='../models/', metavar='DIRPATH',
                        help='directory path to save the model and predictions')
    parser.add_argument('--resume-from', type=str, default=None, metavar='PATH',
                        help='checkpoint path to resume from')

    train(**vars(parser.parse_args()))
