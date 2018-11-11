"""
Misc functions to be settled
"""

from __future__ import division
import xml.dom.minidom
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    rbf_kernel
from sklearn.utils.validation import check_array

import acepy.index.multi_label_tools


def check_matrix(matrix):
    """check if the given matrix is legal."""
    matrix = check_array(matrix, accept_sparse='csr', ensure_2d=False, order='C')
    if matrix.ndim != 2:
        if matrix.ndim == 1 and len(matrix) == 1:
            matrix = matrix.reshape(1, -1)
        else:
            raise TypeError("Matrix should be a 2D array with [n_samples, n_features] or [n_samples, n_classes].")
    return matrix


def get_gaussian_kernel_mat(X, sigma=1.0, check_arr=True):
    """Calculate kernel matrix between X and X.

    Parameters
    ----------
    X: np.ndarray
        data matrix with [n_samples, n_features]

    sigma: float, optional (default=1.0)
        the width in gaussian kernel.

    check_arr: bool, optional (default=True)
        whether to check the given feature matrix.

    Returns
    -------
    K: np.ndarray
        Kernel matrix between X and X.
    """
    if check_arr:
        X = check_array(X, accept_sparse='csr', ensure_2d=True, order='C')
    else:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
    n = X.shape[0]
    tmp = np.sum(X ** 2, axis=1).reshape(1, -1)
    return np.exp((-tmp.T.dot(np.ones((1, n))) - np.ones((n, 1)).dot(tmp) + 2 * (X.dot(X.T))) / (2 * (sigma ** 2)))


def randperm(n, k=None):
    """Generate a random array which contains k elements range from (n[0]:n[1])

    Parameters
    ----------
    n: int or tuple
        range from [n[0]:n[1]], include n[0] and n[1].
        if an int is given, then n[0] = 0

    k: int, optional (default=end - start + 1)
        how many numbers will be generated. should not larger than n[1]-n[0]+1,
        default=n[1] - n[0] + 1.

    Returns
    -------
    perm: list
        the generated array.
    """
    if isinstance(n, np.generic):
        n = np.asscalar(n)
    if isinstance(n, tuple):
        if n[0] is not None:
            start = n[0]
        else:
            start = 0
        end = n[1]
    elif isinstance(n, int):
        start = 0
        end = n
    else:
        raise TypeError("n must be tuple or int.")

    if k is None:
        k = end - start + 1
    if not isinstance(k, int):
        raise TypeError("k must be an int.")
    if k > end - start + 1:
        raise ValueError("k should not larger than n[1]-n[0]+1")

    randarr = np.arange(start, end + 1)
    np.random.shuffle(randarr)
    return randarr[0:k]


def _is_arraylike(x):
    """Returns whether the input is array-like"""
    return (hasattr(x, '__len__') or
            hasattr(x, 'shape') or
            hasattr(x, '__array__'))


def nlargestarg(a, n):
    """Return n largest values' indexes of the given array a.

    Parameters
    ----------
    a: array
        Data array.

    n: int
        The number of returned args.

    Returns
    -------
    nlargestarg: list
        The n largest args in array a.
    """
    assert(_is_arraylike(a))
    assert (n > 0)
    argret = np.argsort(a)
    # ascend
    return argret[argret.size - n:]


def nsmallestarg(a, n):
    """Return n smallest values' indexes of the given array a.

    Parameters
    ----------
    a: array
        Data array.

    n: int
        The number of returned args.

    Returns
    -------
    nlargestarg: list
        The n smallest args in array a.
    """
    assert(_is_arraylike(a))
    assert (n > 0)
    argret = np.argsort(a)
    # ascend
    return argret[0:n]


def calc_kernel_matrix(X, kernel, **kwargs):
    """calculate kernel matrix between X and X.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    Returns
    -------

    """
    if kernel == 'rbf':
        K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
    elif kernel == 'poly':
        K = polynomial_kernel(X=X,
                              Y=X,
                              coef0=kwargs.pop('coef0', 1),
                              degree=kwargs.pop('degree', 3),
                              gamma=kwargs.pop('gamma', 1.))
    elif kernel == 'linear':
        K = linear_kernel(X=X, Y=X)
    elif hasattr(kernel, '__call__'):
        K = kernel(X=np.array(X), Y=np.array(X))
    else:
        raise NotImplementedError

    return K


def check_one_to_one_correspondence(*args):
    """Check if the parameters are one-to-one correspondence.

    Parameters
    ----------
    args: object
        The parameters to test.

    Returns
    -------
    result: int
        Whether the parameters are one-to-one correspondence.
        1 : yes
        0 : no
        -1: some parameters have the length 1.
    """
    first_not_none = True
    result = True
    for item in args:
        # only check not none object
        if item is not None:
            if first_not_none:
                # record item type
                first_not_none = False
                if_array = isinstance(item, (list, np.ndarray))
                if if_array:
                    itemlen = len(item)
                else:
                    itemlen = 1
            else:
                if isinstance(item, (list, np.ndarray)):
                    if len(item) != itemlen:
                        return False
                else:
                    if itemlen != 1:
                        return False
    return True


def unpack(*args):
    """Unpack the list with only one element."""
    ret_args = []
    for arg in args:
        if isinstance(arg, (list, np.ndarray)):
            if len(arg) == 1:
                ret_args.append(arg[0])
            else:
                ret_args.append(arg)
        else:
            ret_args.append(arg)
    return tuple(ret_args)


if __name__ == '__main__':
    # a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # print(get_gaussian_kernel_mat(a))
    # print()
    # lm = np.random.randn(2, 4)
    # print(lm)
    # print(multi_label_tools.get_labelmatrix_in_multilabel([(0,), (1, 1), (1, 2)], lm))
    # print(multi_label_tools.get_labelmatrix_in_multilabel([(1, (0, 1)), (0, [1, 2]), (1, 2)], lm))
    pass