"""
Implement several selected state-of-the-art methods.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings
import copy
import pickle
import sys

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, \
    rbf_kernel

import acepy.utils.interface
import acepy.utils.misc


class QueryInstanceQUIRE(acepy.utils.interface.BaseIndexQuery):
    """Querying Informative and Representative Examples (QUIRE)

    Query the most informative and representative examples where the metrics
    measuring and combining are done using min-max approach.

    The implementation refers to the project: https://github.com/ntucllab/libact

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    train_idx: array-like
        the index of training data.

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


    Attributes
    ----------

    Examples
    --------


    References
    ----------
    [1] Yang, Y.-Y.; Lee, S.-C.; Chung, Y.-A.; Wu, T.-E.; Chen, S.-
        A.; and Lin, H.-T. 2017. libact: Pool-based active learning
        in python. Technical report, National Taiwan University.
        available as arXiv preprint https://arxiv.org/abs/
        1710.00379.

    [2] Huang, S.; Jin, R.; and Zhou, Z. 2014. Active learning by
        querying informative and representative examples. IEEE
        Transactions on Pattern Analysis and Machine Intelligence
        36(10):1936–1949
    """

    def __init__(self, X, y, train_idx, **kwargs):
        # K: kernel matrix
        #
        X = np.asarray(X)[train_idx]
        y = np.asarray(y)[train_idx]
        self._train_idx = train_idx

        self.y = np.array(y)
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
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))

    def select(self, label_index, unlabel_index, batch_size=1):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        # build map from value to index
        label_index_in_train = [np.where(self._train_idx == i)[0][0] for i in label_index]
        unlabel_index_in_train = [np.where(self._train_idx == i)[0][0] for i in unlabel_index]
        # end

        L = self.L
        Lindex = list(label_index_in_train)
        Uindex = list(unlabel_index_in_train)
        query_index = -1
        min_eva = np.inf
        # y_labeled = np.array([label for label in self.y if label is not None])
        y_labeled = self.y[Lindex]
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                                                          np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled,
            )
            eva = L[each_index][each_index] - \
                  det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)

            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        return [self._train_idx[query_index]]


class QueryInstanceGraphDensity(acepy.utils.interface.BaseIndexQuery):
    """Diversity promoting sampling method that uses graph density to determine
    most representative points.

    The implementation refers to the https://github.com/google/active-learning

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    train_idx: array-like
        the index of training data.

    metric: str, optional (default='manhattan')
        the distance metric.
        valid metric = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                      'braycurtis', 'canberra', 'chebyshev', 'correlation',
                      'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                      'russellrao', 'seuclidean', 'sokalmichener',
                      'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]

    References
    ----------
    [1] Ebert, S.; Fritz, M.; and Schiele, B. 2012. RALF: A reinforced
        active learning formulation for object class recognition.
        In 2012 IEEE Conference on Computer Vision and  Pattern Recognition,
        Providence, RI, USA, June 16-21, 2012,
        3626–3633.
    """

    def __init__(self, X, y, train_idx, metric='manhattan'):
        self.metric = metric
        super(QueryInstanceGraphDensity, self).__init__(X, y)
        # Set gamma for gaussian kernel to be equal to 1/n_features
        self.gamma = 1. / self.X.shape[1]
        self.flat_X = X[train_idx, :]
        self.train_idx = train_idx
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._compute_graph_density()

    def _compute_graph_density(self, n_neighbor=10):
        # kneighbors graph is constructed using k=10
        connect = kneighbors_graph(self.flat_X, n_neighbor, p=1)
        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()
        inds = zip(neighbors[0], neighbors[1])
        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]
            distance = pairwise_distances(self.flat_X[[i]], self.flat_X[[j]], metric=self.metric)
            distance = distance[0, 0]
            weight = np.exp(-distance * self.gamma)
            connect[i, j] = weight
            connect[j, i] = weight
        self.connect = connect
        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.
        self.graph_density = np.zeros(self.flat_X.shape[0])
        for i in np.arange(self.flat_X.shape[0]):
            self.graph_density[i] = connect[i, :].sum() / (connect[i, :] > 0).sum()
        self.starting_density = copy.deepcopy(self.graph_density)

    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        # If a neighbor has already been sampled, reduce the graph density
        # for its direct neighbors to promote diversity.
        batch = set()

        # build map from value to index
        try:
            label_index_in_train = [self.train_idx.index(i) for i in label_index]
        except:
            label_index_in_train = [np.where(self.train_idx == i)[0][0] for i in label_index]
        # end

        self.graph_density[label_index_in_train] = min(self.graph_density) - 1
        selected_arr = []
        while len(batch) < batch_size:
            selected = np.argmax(self.graph_density)
            selected_arr.append(selected)
            neighbors = (self.connect[selected, :] > 0).nonzero()[1]
            self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
            assert (self.train_idx[selected] in unlabel_index)
            batch.add(self.train_idx[selected])
            self.graph_density[label_index_in_train] = min(self.graph_density) - 1
            # self.graph_density[list(batch)] = min(self.graph_density) - 1
            self.graph_density[selected_arr] = min(self.graph_density) - 1
        return list(batch)

    def to_dict(self):
        output = {}
        output['connectivity'] = self.connect
        output['graph_density'] = self.starting_density
        return output

