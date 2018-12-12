"""
Implement query strategies for multi-label setting.

There are 2 categories of methods.
1. Query instance-label pairs: QUIRE (TPAMI’14), cosMAL (ICIP’15), Random

2. Query all labels of an instance: MMC (KDD’09), Adaptive (IJCAI’13), Random
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import pickle
import sys
import warnings

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    rbf_kernel
from sklearn.neighbors import kneighbors_graph

from .base import BaseMultiLabelQuery
from ..index.index_collections import MultiLabelIndexCollection
from ..utils.misc import nsmallestarg, nlargestarg

class _LabelRankingModel:
    """Label ranking model is a classification model in multi-label setting.
    It combines label ranking with threshold learning, and use SGD to optimize.

    It accept 3 types of labels:
    >0: the higher the more relevant
    <0: the lower the less irrelevant
    0 : unknown (not use this label when updating)

    This class mainly used for AURO and AUDI method for multi label querying.
    """


class QueryMultiLabelQUIRE(BaseMultiLabelQuery):
    """QUIRE will select an instance-label pair based on the
    informativeness and representativeness for multi-label setting.

    This method will train a multi label classification model by combining
    label ranking with threshold learning and use it to evaluate the unlabeled data.
    Thus it is no need to pass any model.

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.

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

    References
    ----------
    [1] Huang, S.; Jin, R.; and Zhou, Z. 2014. Active learning by
        querying informative and representative examples. IEEE
        Transactions on Pattern Analysis and Machine Intelligence
        36(10):1936–1949
    """

    def __init__(self, X, y, **kwargs):
        # K: kernel matrix
        super(QueryMultiLabelQUIRE, self).__init__(X, y)
        self.lmbda = kwargs.pop('lambda', 1.)
        self.kernel = kwargs.pop('kernel', 'rbf')
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=kwargs.pop('coef0', 1),
                                       degree=kwargs.pop('degree', 3),
                                       gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self.K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))
        self._nsamples, self._nclass = self.y.shape
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))

    def select(self, label_index, unlabel_index, **kwargs):
        if len(unlabel_index) <= 1:
            return unlabel_index
        unlabel_index = list(self._check_multi_label_ind(unlabel_index))
        label_index = list(self._check_multi_label_ind(label_index))

        nU = len(unlabel_index)
        # Only use the 2nd element
        Uidx = [uind[1] for uind in unlabel_index]
        Sidx = [lind[1] for lind in label_index]
        Ys = [self.y[mlab_ind] for mlab_ind in unlabel_index]
        Luu = self.L[np.ix_(Uidx, Uidx)]
        Lsu = self.L[np.ix_(Sidx, Uidx)]
        LL = np.linalg.inv(Luu)
        # calculate the evaluation value for each pair in U
        vals = np.zeros(nU, 1)
        YsLsu = np.dot(Ys, Lsu)
        for i in range(nU):
            tmpidx = list(range(nU))
            tmpidx.remove(i)
            Lqq = Luu[i, i]
            Lqr = Luu[i, tmpidx]
            tmp0 = Lqq  # +Ys'*Lss*Ys;

            b = -(LL[i, tmpidx])
            invLrr = LL[np.ix_(tmpidx, tmpidx)] - b.T.dot(b)/LL[i,i]
            vt1 = YsLsu[:, tmpidx]
            vt2 = 2 * YsLsu[:, i]
            tmp1 = vt1 + Lqr
            tmp1 = vt2 - tmp1.dot(invLrr).dot(tmp1.T)
            tmp2 = vt1 - Lqr
            tmp2 = -vt2 - tmp2.dot(invLrr).dot(tmp2.T)
            vals[i] = np.max((tmp0 + tmp1), (tmp0 + tmp2))

        idx_selected = nsmallestarg(vals, 1)[0]
        return unlabel_index[idx_selected]

class QueryMultiLabelAUDI(BaseMultiLabelQuery):
    """AUDI select an instance-label pair based on Uncertainty and Diversity.

    This method will train a multilabel classification model by combining
    label ranking with threshold learning and use it to evaluate the unlabeled data.
    Thus it is no need to pass any model.

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
    """
