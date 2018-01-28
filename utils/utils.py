import sys
import time
import numpy as np

from tqdm import tqdm, tqdm_notebook
def _is_in_ipython():
    try: __IPYTHON__; return True
    except NameError: return False
_t = tqdm_notebook if _is_in_ipython() else tqdm


class Stopwatch(object):
    """
    A simple cross-platform
    context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as s:
    ...     time.sleep(0.1) # doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    >>> with Stopwatch(verbose=False) as s:
    ...     time.sleep(0.1)
    >>> import math
    >>> math.fabs(s.elapsed() - 0.1) < 0.05
    True
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        if sys.platform == 'win32':
            # on Windows, the best timer is time.clock()
            self._timer_func = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_func = time.time
        self.reset()

    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self.verbose:
            print "Elapsed time: {0:.3f} sec".format(self.elapsed())

    def start(self):
        if not self._is_running:
            self._start = self._timer_func()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total += (self._timer_func() - self._start)
            self._is_running = False
        return self

    def elapsed(self):
        if self._is_running:
            now = self._timer_func()
            self._total += (now - self._start)
            self._start = now
        return self._total

    def reset(self):
        self._start = 0.
        self._total = 0.
        self._is_running = False
        return self


def progress_iter(iterable, verbose=False, *progress_args, **progress_kwargs):
    if verbose: iterable = _t(iterable, total=len(iterable),
                              *progress_args, **progress_kwargs)
    for item in iterable:
        yield item

def batch_iter(X, batch_size=10):
    """
    Examples
    --------
    >>> X = np.arange(36).reshape((12, 3))
    >>> for X_b in batch_iter(X, batch_size=5):
    ...     print X_b
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]]
    [[15 16 17]
     [18 19 20]
     [21 22 23]
     [24 25 26]
     [27 28 29]]
    [[30 31 32]
     [33 34 35]]
    """
    N = len(X)
    start = 0
    while start < N:
        yield X[start:start + batch_size]
        start += batch_size

def write_during_training(s):
    tqdm.write(s)

def print_inline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def softmax(z):
    """
    Examples
    --------
    >>> z = np.log([1, 2, 5])
    >>> softmax(z)
    array([[ 0.125,  0.25 ,  0.625]])
    >>> z += 100.
    >>> softmax(z)
    array([[ 0.125,  0.25 ,  0.625]])
    """
    z = np.atleast_2d(z)
    # avoid numerical overflow by subtracting max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    y = e / np.sum(e, axis=1, keepdims=True)
    return y

def inv_softmax(p):
    """
    Since inverse of softmax function is not unique,
    this function returns `z` such as z.min(axis=1) == 0 (zero vector)

    Examples
    --------
    >>> p = softmax(np.arange(3 * 4).reshape((3, 4)))
    >>> z = inv_softmax(p)
    >>> z
    array([[ 0.,  1.,  2.,  3.],
           [ 0.,  1.,  2.,  3.],
           [ 0.,  1.,  2.,  3.]])
    >>> inv_softmax(softmax(np.arange(10.)))
    array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
    """
    p = np.atleast_2d(p)
    p = np.clip(p, 1e-8, 1.)
    z = np.log(p)
    z -= z.min(axis=1)[:, np.newaxis]
    return z

def one_hot(y):
    """Convert `y` to one-hot encoding.

    Examples
    --------
    >>> y = [2, 1, 0, 2, 0]
    >>> one_hot(y)
    array([[ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]])
    """
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]

def one_hot_decision_function(y):
    """
    Examples
    --------
    >>> y = [[0.1, 0.4, 0.5],
    ...      [0.8, 0.1, 0.1],
    ...      [0.2, 0.2, 0.6],
    ...      [0.3, 0.4, 0.3]]
    >>> one_hot_decision_function(y)
    array([[ 0.,  0.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    """
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z

def unhot(y):
    """
    Map `y` from one-hot encoding to {0, ..., `n_classes` - 1}.

    Examples
    --------
    >>> y = [[0, 0, 1],
    ...      [0, 1, 0],
    ...      [1, 0, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 0]]
    >>> unhot(y)
    array([2, 1, 0, 2, 0])
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    _, n_classes = y.shape
    return y.dot(np.arange(n_classes))
