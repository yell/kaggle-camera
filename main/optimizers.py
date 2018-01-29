import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import progress_iter, print_inline, write_during_training


class BaseOptimizerCallback(object):
    def __init__(self):
        self._opt = None

class ReduceLROnPlateau(BaseOptimizerCallback):
    def __init__(self, factor=0.5, patience=10, min_lr=1e-8, eps=1e-6, verbose=True):
        super(ReduceLROnPlateau, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        self._waiting = 0

    def __call__(self):
        if not self._opt.best_val_acc or not self._opt.best_val_acc:
            return
        if max(param_group['lr'] for param_group in self._opt.optim.param_groups) * self.factor < self.min_lr:
            return
        if np.abs(self._opt.best_val_acc - self._opt.val_acc_history[-1]) < self.eps:
            self._waiting = 0
        else:
            self._waiting += 1
        if self._waiting >= self.patience:
            if self.verbose:
                write_during_training("INFO: reducing learning rate!")
            self._opt._mul_lr_by(self.factor)
            self._waiting = 0


class ClassificationOptimizer(object):
    """
    Parameters
    ----------
    optim : `torch.optim`, optional
        Optimization algorithm to use.
    optim_params : dict, optional
        Parameters passed to `optim`.
    loss_func : `torch.nn` loss function, optional
        Loss function to use.
    max_epoch : int, optional
        Number of epochs to train.
    use_cuda : {bool, None}, optional
        Whether to compute on GPU.
        If None, CUDA availability is checked.
    verbose : bool, optional
        Whether to print progress.
    save_every_epoch : non-negative int, optional
        If positive, saves model according to `path_template`
        every specified number of epochs. Setting to zero
        disables this feature.
    path_template : str, optional
        Path template for model to be saved. Actual path is:
        ```
        path_template.format(acc=..., loss=..., epoch=...) + '.ckpt'
        ```
    callbacks : `BaseOptimizerCallback` iterable
    """
    def __init__(self, model, model_params=None, optim=None, optim_params=None,
                 loss_func=nn.CrossEntropyLoss(), max_epoch=10, val_each_epoch=1,
                 cyclic_lr=None, use_cuda=None, verbose=True, path_template='{acc:.4f}-{epoch}',
                 callbacks=None):
        self.model = model
        if model_params is None or not len(model_params):
            model_params = filter(lambda x: x.requires_grad, self.model.parameters())

        optim = optim or torch.optim.SGD
        optim_params = optim_params or {}
        optim_params.setdefault('lr', 1e-3)
        self.optim = optim(model_params, **optim_params)

        self.loss_func = loss_func
        self.max_epoch = max_epoch
        self.val_each_epoch = val_each_epoch
        self.cyclic_lr = cyclic_lr
        self.m, self.M, self.stepsize = None, None, None
        if self.cyclic_lr:
            self.m, self.M, self.stepsize = self.cyclic_lr

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model.cuda()

        self.verbose = verbose
        self.path_template = path_template
        self.callbacks = callbacks
        if hasattr(self.callbacks, '__iter__'):
            for cb in self.callbacks:
                cb._opt = self

        self.dirpath, _ = os.path.split(self.path_template)
        if self.dirpath and not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.epoch = 0

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

        self.best_val_acc = None
        self.path = None
        self.best_path = None
        self.is_best = False

    def _rm(self, path):
        if path and os.path.isfile(path):
            os.remove(path)

    def _save(self, path):
        state = {
            'epoch': self.epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'train_loss': self.train_loss_history,
            'train_acc': self.train_acc_history,
            'val_loss': self.val_loss_history,
            'val_acc': self.val_acc_history,
            'best_val_acc': self.best_val_acc
        }
        torch.save(state, path)

    def _mul_lr_by(self, lrm):
        if not hasattr(lrm, '__iter__'):
            lrm = [lrm]
        if len(lrm) < len(list(self.optim.param_groups)):
            lrm = [lrm[0]] * len(list(self.optim.param_groups))
        for i, param_group in enumerate(self.optim.param_groups):
            param_group['lr'] *= lrm[i]

    def _get_cyclic_lr(self, epoch):
        """
        Reference
        ---------
        https://arxiv.org/pdf/1506.01186.pdf
        """
        cycle = np.floor(1. + 0.5 * epoch / float(self.stepsize))
        x = np.abs(epoch / float(self.stepsize) - 2. * cycle + 1.)
        lr = self.m + (self.M - self.m) * max(0, 1. - x)
        return lr

    def _get_cyclic_lrm(self):
        return self._get_cyclic_lr(self.epoch) / self._get_cyclic_lr(self.epoch - 1)

    def save(self, is_best):
        if is_best or (not is_best and self.path != self.best_path):
            self._rm(self.path)

        if self.val_acc_history:
            self.path = self.path_template.format(acc=self.val_acc_history[-1],
                                                  loss=self.val_loss_history[-1],
                                                  epoch=self.epoch)
        else:
            self.path = '1'
        if not self.path.endswith('.ckpt'):
            self.path += '.ckpt'

        if is_best:
            self._rm(self.best_path)
            self.best_path = self.path

        self._save(self.path)
        return self

    def load(self, path):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state'])
            self.optim.load_state_dict(checkpoint['optim_state'])
            self.train_loss_history = checkpoint['train_loss']
            self.train_acc_history = checkpoint['train_acc']
            self.val_loss_history = checkpoint['val_loss']
            self.val_acc_history = checkpoint['val_acc']
            self.best_val_acc = checkpoint['best_val_acc']
            return self
        elif os.path.isdir(path):
            # if directory is provided, load latest checkpoint
            ckpt_paths = glob.glob(os.path.join(path, '*.ckpt'))
            ckpt_path = max(ckpt_paths, key=os.path.getctime)
            print "INFO: Loading checkpoint '{0}'".format(ckpt_path)
            return self.load(ckpt_path)
        else:
            raise IOError('invalid checkpoint path: \'{0}\''.format(path))

    def train_epoch(self, train_loader):
        self.model.train()

        epoch_iter = 0
        epoch_train_loss = 0.
        epoch_correct = 0
        epoch_total = 0
        epoch_acc = 0.
        epoch_train_loss_history = []

        for X_batch, y_batch in progress_iter(iterable=train_loader, verbose=self.verbose,
                                              leave=True, ncols=64, desc='epoch'):
            X_batch, manip = X_batch
            if self.use_cuda:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                manip = manip.cuda()
            X_batch, y_batch = Variable(X_batch), Variable(y_batch)
            self.optim.zero_grad()
            out = self.model((X_batch, Variable(manip)))

            loss = self.loss_func(out, y_batch)
            epoch_train_loss_history.append( loss.data[0] )
            epoch_train_loss *= epoch_iter / (epoch_iter + 1.)
            epoch_train_loss += loss.data[0] / (epoch_iter + 1.)
            epoch_iter += 1

            _, y_pred = torch.max(out.data, 1)
            epoch_correct += y_pred.eq(y_batch.data).cpu().sum()
            epoch_total += y_batch.size(0)
            epoch_acc = epoch_correct/float(epoch_total)

            if self.verbose:
                s = "loss: {0:.4f} acc: {1:.4f}".format(epoch_train_loss, epoch_acc)
                print_inline(s)

            loss.backward()
            self.optim.step()

        # update global history
        self.train_loss_history.append( epoch_train_loss_history )
        self.train_acc_history.append( epoch_acc )

        # update cyclic LR if enabled
        if self.cyclic_lr:
            lrm = self._get_cyclic_lrm()
            self._mul_lr_by(lrm)

    def test(self, test_loader, validation=False):
        self.model.eval()

        outs = []
        test_loss_history = []
        correct = 0
        total = 0

        for X_batch, y_batch in progress_iter(iterable=test_loader, verbose=self.verbose,
                                              leave=False, ncols=64,
                                              desc='validation' if validation else 'predicting'):
            X_batch, manip = X_batch
            if len(X_batch.size()) > 4:
                bs, tta_n, c, h, w = X_batch.size()
                X_batch = X_batch.view(-1, c, h, w)
                y_batch = y_batch.repeat(tta_n)
                manip = manip.repeat(tta_n, 0)
            if self.use_cuda:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                manip = manip.cuda()
            X_batch, y_batch = Variable(X_batch, volatile=True), Variable(y_batch, volatile=True)
            out = self.model((X_batch, Variable(manip, volatile=True)))
            outs.append(out.data.cpu().numpy())
            test_loss_history.append( self.loss_func(out, y_batch).data[0] )

            _, y_pred = torch.max(out.data, 1)
            correct += y_pred.eq(y_batch.data).cpu().sum()
            total += y_batch.size(0)

        if validation:
            val_acc = correct/float(total)
            val_loss = test_loss_history

            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(test_loss_history)

            if self.verbose:
                s = "epoch: {0:{1}}/{2}".format(self.epoch, len(str(self.max_epoch)), self.max_epoch)
                s += "; val.acc: {0:.4f}".format(val_acc)
                s += "; val.loss: {0:.4f}".format(np.mean(val_loss))
                write_during_training(s + '\n')

        return outs, test_loss_history

    def train(self, train_loader, val_loader):
        for self.epoch in progress_iter(iterable=xrange(self.epoch + 1, self.max_epoch + 1),
                                        verbose=self.verbose, leave=False, ncols=72, desc='training'):
            self.train_epoch(train_loader)
            if self.epoch % self.val_each_epoch == 0:
                self.test(val_loader, validation=True)

            self.is_best = False
            if self.best_val_acc is None or self.val_acc_history[-1] > self.best_val_acc:
                if self.val_acc_history:
                    self.best_val_acc = self.val_acc_history[-1]
                    self.is_best = True
            self.save(self.is_best)

            if self.epoch % self.val_each_epoch == 0:
                for cb in self.callbacks:
                    cb()
