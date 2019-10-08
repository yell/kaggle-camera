import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def latest_ckpt_learning_curves(dirpath, **kwargs):
    ckpts = glob.glob(os.path.join(dirpath, '*.ckpt'))
    last_ckpt = max(ckpts, key=os.path.getctime)
    return ckpt_learning_curves(last_ckpt, **kwargs)

def ckpt_learning_curves(ckpt_path, b=8, gamma=0., **plt_learning_curves_kwargs):
    checkpoint = torch.load(ckpt_path, map_location={'cuda:0': 'cpu'})

    train_loss_history = checkpoint['train_loss']
    train_acc_history = checkpoint['train_acc']
    val_loss_history = [np.mean(v) for v in checkpoint['val_loss']]
    val_acc_history = checkpoint['val_acc']

    n = (len(train_loss_history) / b)
    train_loss_history = train_loss_history[:b * n]
    train_acc_history = train_acc_history[:b * n]
    val_loss_history = val_loss_history[:n]
    val_acc_history = val_acc_history[:n]

    train_loss_history = [np.mean(v) for v in train_loss_history]
    train_loss_history = np.asarray(train_loss_history).reshape(-1, b).mean(axis=1).tolist()
    # train_loss_history = [[v] for v in train_loss_history]
    train_acc_history = np.asarray(train_acc_history).reshape(-1, b).mean(axis=1).tolist()

    train_loss_history = [[v] for v in ema(train_loss_history, gamma)]
    train_acc_history = ema(train_acc_history, gamma)
    val_loss_history = ema(val_loss_history, gamma)
    val_acc_history = ema(val_acc_history, gamma)

    plt_learning_curves_kwargs.setdefault('min_acc', 0.1)
    plt_learning_curves_kwargs.setdefault('filepath', 'learning_curves.png')
    # noinspection PyTypeChecker
    plot_learning_curves(train_loss_history, train_acc_history,
                         val_loss_history, val_acc_history, **plt_learning_curves_kwargs)
    return val_acc_history[-1]

def ema(w, gamma=0.9):
    w_ema = [w[0]]
    for i in xrange(1, len(w)):
        w_ema.append(gamma * w_ema[-1] + (1. - gamma) * w[i])
    return w_ema

def plot_learning_curves(l, a, vl, va, last_epochs=None, min_loss=0., max_loss=25., min_acc=0.4, filepath=None):
    n_batches = len(l[0])
    n_epochs = len(l)
    x = np.linspace(1., n_epochs, n_epochs, endpoint=True)
    z = np.linspace(1., n_epochs, (n_epochs - 1) * n_batches, endpoint=True)
    if last_epochs:
        l = l[-last_epochs:]
        a = a[-last_epochs:]
        vl = vl[-last_epochs:]
        va = va[-last_epochs:]
        x = x[-last_epochs:]
        z = z[-((last_epochs - 1) * n_batches):]

    plt.close("all")

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()

    yl = ax.get_ygridlines()
    for yl_ in yl:
        yl_.set_color('w')

    l_mean = [np.mean(l_) for l_ in l]
    marker = 'o' if len(vl) < 100 else None
    L1 = ax.plot(z, np.concatenate(l[1:]), color='#5a053f', lw=2, label='training loss')
    # ax.plot(x, l_mean, color='r', lw=2, marker='o', label='training loss mean')
    L2 = ax.plot(x, vl, color='#e6155a', lw=2, marker=marker, label='validation loss')
    ax.set_ylim([min_loss, min(max_loss, max(max(max(l[1:])), max(vl[1:])))])
    ax.set_xlim([1, n_epochs])

    L3 = ax2.plot(x, a, color='#124f90', lw=2, marker=marker, label='training accuracy')
    L4 = ax2.plot(x, va, color='#6dbb30', lw=2, marker=marker, label='validation accuracy')
    ax2.set_ylim([max(min_acc, min(min(a), min(va))), 1.])

    ax2.spines['left'].set_color('black')
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_color('black')
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['bottom'].set_color('black')
    ax2.spines['bottom'].set_linewidth(2)

    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)

    ax.set_title('Learning curves', fontsize=27, weight='bold', y=1.03)

    ax.set_ylabel('loss', fontsize=23)
    ax2.set_ylabel('accuracy', fontsize=23)
    ax.set_xlabel('epoch', fontsize=23)

    Ls = L1 + L2 + L3 + L4
    labs = [l.get_label() for l in Ls]
    leg = plt.legend(Ls, labs, loc='lower left', fontsize=18, frameon=True)
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(2)

    if filepath:
        if not filepath.endswith('.png'):
            filepath += '.png'
        fig.savefig(filepath, dpi=fig.dpi)
        plt.close(fig)


if __name__ == '__main__':
    ckpt_learning_curves('../kaggle-camera/models/0.8826-1384.ckpt')
