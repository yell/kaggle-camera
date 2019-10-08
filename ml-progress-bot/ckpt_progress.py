import os
import sys
import glob
from datetime import datetime


def get_progress(dirpath='', ckpt_ext='*.ckpt'):
    dirpath = dirpath or 'models'
    d = {}
    for dpath in os.listdir(dirpath):
        ckpts = glob.glob(os.path.join(dirpath, dpath, ckpt_ext))
        if ckpts:
            ckpts.sort(reverse=True)
            best_ckpt = os.path.split(ckpts[0])[-1]
            last_ckpt = max(ckpts, key=os.path.getctime)
            modified_time = datetime.fromtimestamp(os.path.getctime(last_ckpt)).strftime('%d.%m %H:%M:%S')
            last_ckpt = os.path.split(last_ckpt)[-1]
            d['*{0}*'.format(dpath)] = '{0}, *{1}*, {2}'.format(modified_time, best_ckpt, last_ckpt)
        else:
            d['*{0}*'.format(dpath)] = '<no checkpoints>'
    return d


if __name__ == '__main__':
    get_progress(sys.argv[1])
