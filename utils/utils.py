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


def print_inline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


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

if __name__ == '__main__':
    # run corresponding tests
    from testing import run_tests
    run_tests(__file__)
