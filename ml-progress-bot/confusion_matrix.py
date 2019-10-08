import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_validation_cm(dirpath):
    y_val = np.load(os.path.join(dirpath, 'y_val.npy')).flatten()
    y_pred = np.load(os.path.join(dirpath, 'y_pred.npy'))
    assert len(y_val) == len(y_pred)

    C = confusion_matrix(y_val, y_pred)

    m = (y_val == y_pred)
    per_class_acc = [100. * sum(m * (y_val == i)) / float(sum((y_val == i))) for i in xrange(10)]
    per_class_acc = map(lambda s: "{:04.1f}".format(s), per_class_acc)
    title = '   '.join(per_class_acc)

    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=19.5)
    plot_confusion_matrix(C, labels=['HTC-1-M7',
                                     'LG-Nexus-5x',
                                     'Motorola-Droid-Maxx',
                                     'Motorola-Nexus-6',
                                     'Motorola-X',
                                     'Samsung-Galaxy-Note3',
                                     'Samsung-Galaxy-S4',
                                     'Sony-NEX-7',
                                     'iPhone-4s',
                                     'iPhone-6'])
    plt.savefig('cm.png', bbox_inches='tight')


def plot_confusion_matrix(C, labels=None, labels_fontsize=13, **heatmap_params):
    # default params
    labels = labels or range(C.shape[0])
    annot_fontsize = 14
    xy_label_fontsize = 21

    # set default params where possible
    if not 'annot' in heatmap_params:
        heatmap_params['annot'] = True
    if not 'fmt' in heatmap_params:
        heatmap_params['fmt'] = 'd' if C.dtype is np.dtype('int') else '.3f'
    if not 'annot_kws' in heatmap_params:
        heatmap_params['annot_kws'] = {'size': annot_fontsize}
    elif not 'size' in heatmap_params['annot_kws']:
        heatmap_params['annot_kws']['size'] = annot_fontsize
    if not 'xticklabels' in heatmap_params:
        heatmap_params['xticklabels'] = labels
    if not 'yticklabels' in heatmap_params:
        heatmap_params['yticklabels'] = labels

    # plot the stuff
    with plt.rc_context(rc={'xtick.labelsize': labels_fontsize,
                            'ytick.labelsize': labels_fontsize}):
        ax = sns.heatmap(C, **heatmap_params)
        plt.xlabel('predicted', fontsize=xy_label_fontsize)
        plt.ylabel('actual', fontsize=xy_label_fontsize)
        return ax


if __name__ == '__main__':
    plot_validation_cm('../kaggle-camera/models/test/')
