"""
Implement query strategies for multi-label setting.

There are 2 categories of methods.
1. Query instance-label pairs: QUIRE (TPAMI’14), AUDI (ICDM’13), Random

2. Query all labels of an instance: MMC (KDD’09), Adaptive (IJCAI’13), Random
"""

# Authors: Ying-Peng Tang and Guo-Xiang Li
# License: BSD 3 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import copy
import numpy as np

from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from ..index import IndexCollection, MultiLabelIndexCollection
from ..index.multi_label_tools import get_Xy_in_multilabel
from ..utils.misc import randperm
from .base import BaseIndexQuery, BaseMultiLabelQuery


class _LabelRankingModel_MatlabVer:
    """Label ranking model is a classification model in multi-label setting.
    It combines label ranking with threshold learning, and use SGD to optimize.

    This class is implemented strictly according to the matlab code provided
    by the author. So it's hard to use, but it guarantees correctness.

    Parameters
    ----------
    init_X: 2D array
        Feature matrix of the initial data for training.
        Shape is n*d, one row for an instance with d features.

    init_y: 2D array
        Label matrix of the initial data for training.
        Shape is n*n_classes, one row for an instance, -1 means irrelevant,
        a positive value means relevant, the larger, the more relevant.

    References
    ----------
    [1] S.-J. Huang and Z.-H. Zhou. Active query driven by uncertainty and
        diversity for incremental multi-label learning. In Proceedings
        of the 13th IEEE International Conference on Data Mining, pages
        1079–1084, Dallas, TX, 2013.
    """

    def __init__(self, init_X=None, init_y=None):
        self._init_flag = False
        if init_X is not None and init_y is not None:
            assert len(init_X) == len(init_y)
            assert len(np.shape(init_y)) == 2
            self._init_X = np.asarray(init_X)
            self._init_y = np.asarray(init_y)

            if len(np.nonzero(self._init_y == 2.0)[0]) == 0:
                self._init_y = np.hstack((self._init_y, 2 * np.ones((self._init_y.shape[0], 1))))
                # B, V, AB, AV, Anum, trounds, costs, norm_up, step_size0, num_sub, lmbda, avg_begin, avg_size, n_repeat, \
                # max_query = self.init_model_train(self._init_X, self._init_y)
            self._init_flag = True

    def get_BV(self, AB, AV, Anum):
        return (AV / Anum).T.dot(AB / Anum)

    def init_model_train(self, init_data=None, init_targets=None, n_repeat=10):
        if init_data is None:
            init_data = self._init_X
        if init_targets is None:
            init_targets = self._init_y
        init_data = np.asarray(init_data)
        init_targets = np.asarray(init_targets)
        if len(np.nonzero(init_targets == 2.0)[0]) == 0:
            init_targets = np.hstack((init_targets, 2 * np.ones((init_targets.shape[0], 1))))

        tar_sh = np.shape(init_targets)
        d = np.shape(init_data)[1]
        n_class = tar_sh[1]
        # n_repeat = 10
        max_query = math.floor(tar_sh[0] * (tar_sh[1] - 1) / 2)
        D = 200
        num_sub = 5
        norm_up = np.inf
        lmbda = 0
        step_size0 = 0.05
        avg_begin = 10
        avg_size = 5

        costs = 1.0 / np.arange(start=1, stop=n_class * 5 + 1)
        for k in np.arange(start=1, stop=n_class * 5):
            costs[k] = costs[k - 1] + costs[k]

        V = np.random.normal(0, 1 / np.sqrt(d), (D, d))
        B = np.random.normal(0, 1 / np.sqrt(d), (D, n_class * num_sub))
        # import scipy.io as scio
        # ld = scio.loadmat('F:\\alipy_doc\\alipy-additional-methods-source\\multi label\\AURO\\BV_val.mat')
        # V = ld['V']
        # B = ld['B']

        for k in range(d):
            tmp1 = V[:, k]
            if np.all(tmp1 > norm_up):
                V[:, k] = tmp1 * norm_up / np.linalg.norm(tmp1)
        for k in range(n_class * num_sub):
            tmp1 = B[:, k]
            if np.all(tmp1 > norm_up):
                B[:, k] = tmp1 * norm_up / np.linalg.norm(tmp1)

        AB = 0
        AV = 0
        Anum = 0
        trounds = 0

        for rr in range(n_repeat):
            B, V, AB, AV, Anum, trounds = self.train_model(init_data, init_targets, B, V, costs, norm_up, step_size0,
                                                           num_sub, AB, AV, Anum, trounds, lmbda, avg_begin, avg_size)

        return B, V, AB, AV, Anum, trounds, costs, norm_up, step_size0, num_sub, lmbda, avg_begin, avg_size, n_repeat, max_query

    def train_model(self, data, targets, B, V, costs, norm_up, step_size0, num_sub, AB, AV, Anum, trounds,
                    lmbda, average_begin, average_size):
        """targets: 0 unlabeled, 1 positive, -1 negative, 2 dummy, 0.5 less positive"""
        targets = np.asarray(targets)
        # print(np.nonzero(targets == 2.0))
        if len(np.nonzero(targets == 2.0)[0]) == 0:
            targets = np.hstack((targets, 2 * np.ones((targets.shape[0], 1))))
        data = np.asarray(data)
        B = np.asarray(B)
        V = np.asarray(V)

        n, n_class = np.shape(targets)
        row_ind, col_ind = np.nonzero(targets >= 1)
        train_pairs = np.hstack((row_ind.reshape((-1,1)), col_ind.reshape((-1,1))))

        # tmpnums = np.sum(targets >= 1, axis=1)
        # train_pairs = np.zeros((sum(tmpnums), 1))
        # tmpidx = 0

        # for i in range(n):
        #     train_pairs[tmpidx: tmpidx + tmpnums[i]] = i+1
        #     tmpidx = tmpidx + tmpnums[i]

        # targets = targets.T
        # # tp = np.nonzero(targets.flatten() >= 1)
        # # print(tp[0])
        # # print(len(tp[0]))
        # train_pairs = np.hstack(
        #     (train_pairs,
        #      np.reshape([nz % n_class for nz in np.nonzero(targets.flatten(order='F') >= 1)[0]], newshape=(-1, 1))))
        # # train_pairs[np.nonzero(train_pairs[:, 1] == 0)[0], 1] = n_class
        # targets = targets.T

        n = np.shape(train_pairs)[0]

        random_idx = randperm(n - 1)
        # import scipy.io as scio
        # ld = scio.loadmat('F:\\alipy_doc\\alipy-additional-methods-source\\multi label\\AURO\\perm.mat')
        # random_idx = ld['random_idx'].flatten()-1

        for i in range(n):
            idx_ins = int(train_pairs[random_idx[i], 0])
            xins = data[int(idx_ins), :].T
            idx_class = int(train_pairs[random_idx[i], 1])
            if idx_class == n_class-1:
                idx_irr = np.nonzero(targets[idx_ins, :] == -1)[0]
            # elif idx_class == idxPs[idx_ins]:
            #     idx_irr = np.hstack((np.nonzero(targets[idx_ins, :] == -1)[0], int(idxNs[idx_ins]), n_class - 1))
            else:
                idx_irr = np.hstack((np.nonzero(targets[idx_ins, :] == -1)[0], n_class - 1))
            n_irr = len(idx_irr)

            By = B[:, idx_class * num_sub: (idx_class + 1) * num_sub]
            Vins = V.dot(xins)
            fy = np.max(By.T.dot(Vins), axis=0)
            idx_max_class = np.argmax(By.T.dot(Vins), axis=0)
            By = By[:, idx_max_class]
            fyn = np.NINF
            for j in range(n_irr):
                idx_pick = idx_irr[randperm(n_irr - 1, 1)[0]]
                # print(idx_irr, idx_pick)
                Byn = B[:, idx_pick * num_sub: (idx_pick + 1) * num_sub]
                # [fyn, idx_max_pick] = max(Byn.T.dot(Vins),[],1)
                # if Byn == []:
                #     print(0)
                tmp1 = Byn.T.dot(Vins)
                fyn = np.max(tmp1, axis=0)
                idx_max_pick = np.argmax(tmp1, axis=0)

                if fyn > fy - 1:
                    break

            if fyn > fy - 1:
                step_size = step_size0 / (1 + lmbda * trounds * step_size0)
                trounds = trounds + 1
                Byn = B[:, idx_pick * num_sub + idx_max_pick]
                loss = costs[math.floor(n_irr / (j + 1)) - 1]
                tmp1 = By + step_size * loss * Vins
                tmp3 = np.linalg.norm(tmp1)
                if tmp3 > norm_up:
                    tmp1 = tmp1 * norm_up / tmp3
                tmp2 = Byn - step_size * loss * Vins
                tmp3 = np.linalg.norm(tmp2)
                if tmp3 > norm_up:
                    tmp2 = tmp2 * norm_up / tmp3
                V -= step_size * loss * (
                    B[:, [idx_pick * num_sub + idx_max_pick, idx_class * num_sub + idx_max_class]].dot(
                        np.vstack((xins, -xins))))

                norms = np.linalg.norm(V, axis=0)
                idx_down = np.nonzero(norms > norm_up)[0]
                B[:, idx_class * num_sub + idx_max_class] = tmp1
                B[:, idx_pick * num_sub + idx_max_pick] = tmp2
                if idx_down:
                    norms = norms[norms > norm_up]
                    for k in range(len(idx_down)):
                        V[:, idx_down[k]] = V[:, idx_down[k]] * norm_up / norms[k]
            if trounds > average_begin and i % average_size == 0:
                AB = AB + B
                AV = AV + V
                Anum = Anum + 1

        return B, V, AB, AV, Anum, trounds

    def lr_predict(self, BV, data, num_sub):
        BV = np.asarray(BV)
        data = np.asarray(data)

        fs = data.dot(BV)
        n = data.shape[0]
        n_class = int(fs.shape[1] / num_sub)
        pres = np.ones((n, n_class)) * np.NINF
        for j in range(num_sub):
            f = fs[:, j: fs.shape[1]: num_sub]
            assert (np.all(f.shape == pres.shape))
            pres = np.fmax(pres, f)
        labels = -np.ones((n, n_class - 1))
        for line in range(n_class - 1):
            gt = np.nonzero(pres[:, line] > pres[:, n_class - 1])[0]
            labels[gt, line] = 1
        return pres, labels


class LabelRankingModel(_LabelRankingModel_MatlabVer):
    """Label ranking model is a classification model in multi-label setting.
    It combines label ranking with threshold learning, and use SGD to optimize.
    This class re-encapsulate the _LabelRankingModel_MatlabVer class for
    better use.

    It accept 3 types of labels:
    1 : relevant
    0.5 : less relevant
    -1 : irrelevant

    The labels in algorithms mean:
    2 : dummy
    0 : unknown (not use this label when updating)

    This class is mainly used for AURO and AUDI method for multi label querying.

    Parameters
    ----------
    init_X: 2D array, optional (default=None)
        Feature matrix of the initial data for training.
        Shape is n*d, one row for an instance with d features.

    init_y: 2D array, optional (default=None)
        Label matrix of the initial data for training.
        Shape is n*n_classes, one row for an instance, -1 means irrelevant,
        a positive value means relevant, the larger, the more relevant.

    References
    ----------
    [1] S.-J. Huang and Z.-H. Zhou. Active query driven by uncertainty and
        diversity for incremental multi-label learning. In Proceedings
        of the 13th IEEE International Conference on Data Mining, pages
        1079–1084, Dallas, TX, 2013.
    """

    def __init__(self, init_X=None, init_y=None, **kwargs):
        super(LabelRankingModel, self).__init__(init_X, init_y)
        if self._init_flag is True:
            n_repeat = kwargs.pop('n_repeat', 10)
            self._B, self._V, self._AB, self._AV, self._Anum, self._trounds, self._costs, self._norm_up, \
            self._step_size0, self._num_sub, self._lmbda, self._avg_begin, self._avg_size, self._n_repeat, \
            self._max_query = self.init_model_train(self._init_X, self._init_y, n_repeat=n_repeat)

    def fit(self, X, y, n_repeat=10):
        """Train the model from X and y.

        Parameters
        ----------
        X: 2D array, optional (default=None)
            Feature matrix of the whole dataset.

        y: 2D array, optional (default=None)
            Label matrix of the whole dataset.

        n_repeat: int, optional (default=10)
            The number of optimization iterations.
        """
        if self._init_flag is False:
            self._B, self._V, self._AB, self._AV, self._Anum, self._trounds, self._costs, self._norm_up, \
            self._step_size0, self._num_sub, self._lmbda, self._avg_begin, self._avg_size, self._n_repeat, \
            self._max_query = self.init_model_train(X, y, n_repeat=n_repeat)
            self._init_flag = True
        else:
            for i in range(n_repeat):
                self._B, self._V, self._AB, self._AV, self._Anum, self._trounds = self.train_model(
                    X, y, self._B, self._V, self._costs, self._norm_up,
                    self._step_size0, self._num_sub, self._AB, self._AV, self._Anum, self._trounds,
                    self._lmbda, self._avg_begin, self._avg_size)

    def predict(self, X):
        BV = self.get_BV(self._AB, self._AV, self._Anum)
        return self.lr_predict(BV, X, self._num_sub)


class QueryMultiLabelQUIRE(BaseMultiLabelQuery):
    """QUIRE will select an instance-label pair based on the
    informativeness and representativeness for multi-label setting.

    This method will train a multi label classification model by combining
    label ranking with threshold learning and use it to evaluate the unlabeled data.
    Thus it is no need to pass any model.

    NOTE: QUIRE is better to be used with RBF kernel, and usually the performance is
    good even without fine parameter tuning (that is, it is not very sensitive to
    parameter setting, and using default parameter setting is usually fine)

    Warning: QUIRE must NOT be used with linear kernel on non-textual data.

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
                'Kernel should have size (%d, %d)' % (len(X), len(X)))
        self._nsamples, self._nclass = self.y.shape
        self.L = np.linalg.pinv(self.K + self.lmbda * np.eye(len(X)))

    def select(self, label_index, unlabel_index, **kwargs):
        """Select a subset from the unlabeled set, return the selected instance and label.

        Parameters
        ----------
        label_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        Returns
        -------
        selected_ins_lab_pair: list
            A list of tuples that contains the indexes of selected instance-label pairs.
        """

        if len(unlabel_index) <= 1:
            return unlabel_index
        unlabel_index = self._check_multi_label_ind(unlabel_index)
        label_index = self._check_multi_label_ind(label_index)

        L_kr = np.kron(np.eye(self.y.shape[1]), self.L)
        nU = len(unlabel_index)
        # Only use the 2nd element
        # Uidx_col = []
        # Uidx_row = []
        # for uind in unlabel_index_li:
        #     Uidx_col.append(uind[1])
        #     Uidx_row.append((uind[0] + uind[1] * self.X.shape[0]) % self.L.shape[0])
        Uidx = unlabel_index.get_onedim_index(order='F', ins_num=self.X.shape[0])
        Sidx = label_index.get_onedim_index(order='F', ins_num=self.X.shape[0])
        Uidx = np.sort(Uidx)
        Sidx = np.sort(Sidx)
        Ys = [self.y[lind] for lind in label_index]
        # for lind in label_index:
        #     Sidx.append(lind[0])
        #     Ys.append(self.y[lind])
        Ys = np.asarray(Ys)
        Luu = L_kr[np.ix_(Uidx, Uidx)]
        Lsu = L_kr[np.ix_(Sidx, Uidx)]
        LL = np.linalg.pinv(Luu)
        # calculate the evaluation value for each pair in U
        vals = np.zeros(nU)
        YsLsu = Ys.dot(Lsu).T
        for i in range(nU):
            tmpidx = list(range(nU))
            tmpidx.remove(i)
            Lqq = Luu[i, i]
            Lqr = Luu[i, tmpidx]
            tmp0 = Lqq  # +Ys'*Lss*Ys;

            b = -(LL[i, tmpidx])
            invLrr = LL[np.ix_(tmpidx, tmpidx)] - b.T.dot(b) / LL[i, i]
            vt1 = YsLsu[tmpidx]
            vt2 = 2 * YsLsu[i]
            tmp1 = vt1 + Lqr
            tmp1 = vt2 - tmp1.dot(invLrr).dot(tmp1.T)
            tmp2 = vt1 - Lqr
            tmp2 = -vt2 - tmp2.dot(invLrr).dot(tmp2.T)
            vals[i] = np.fmax((tmp0 + tmp1), (tmp0 + tmp2))

        idx_selected = np.argmin(vals)
        idx_ondim = Uidx[idx_selected]
        return [(idx_ondim % self.X.shape[0], idx_ondim // self.X.shape[0])]


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

    initial_labeled_indexes: {list, np.ndarray, IndexCollection}
        The indexes of initially labeled samples. Used for initializing the LabelRanking model.

    References
    ----------
    [1] S.-J. Huang and Z.-H. Zhou. Active query driven by uncertainty and
        diversity for incremental multi-label learning. In Proceedings
        of the 13th IEEE International Conference on Data Mining, pages
        1079–1084, Dallas, TX, 2013.
    """

    def __init__(self, X, y):
        super(QueryMultiLabelAUDI, self).__init__(X, y)
        # if len(np.nonzero(self.y == 2.0)[0]) == 0:
        #     self.y = np.hstack((self.y, 2 * np.ones((self.y.shape[0], 1))))
        self._lr_model = LabelRankingModel()

    def select(self, label_index, unlabel_index, epsilon=0.5, **kwargs):
        """Select a subset from the unlabeled set, return the selected instance and label.

        Parameters
        ----------
        label_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        epsilon: float, optional (default=0.5)
            The threshold to avoid zero-division.

        Returns
        -------
        selected_ins_lab_pair: list
            A list of tuples that contains the indexes of selected instance-label pairs.
        """
        if len(unlabel_index) <= 1:
            return unlabel_index
        unlabel_index = self._check_multi_label_ind(unlabel_index)
        label_index = self._check_multi_label_ind(label_index)

        # select instance by LCI
        W = unlabel_index.get_matrix_mask(mat_shape=self.y.shape, fill_value=1, sparse=False)
        unlab_data, _, data_ind = get_Xy_in_multilabel(index=unlabel_index, X=self.X, y=self.y)
        lab_data, lab_lab, _ = get_Xy_in_multilabel(index=label_index, X=self.X, y=self.y)
        self._lr_model.fit(lab_data, lab_lab)
        pres, labels = self._lr_model.predict(unlab_data)
        avgP = np.mean(np.sum(self.y[label_index.get_unbroken_instances(), :] == 1, axis=1))
        insvals = -np.abs((np.sum(labels == 1, axis=1) - avgP) / np.fmax(np.sum(W[data_ind, :] == 1, axis=1), epsilon))
        selected_ins = np.argmin(insvals)

        # last line in pres is the predict value of dummy label
        # select label by calculating the distance between each label with dummy label

        # set the known entries to inf
        pres_mask = np.asarray(1 - W[data_ind], dtype=bool)
        pres_tmp = pres[:, 0:-1]
        pres_tmp[pres_mask] = np.NINF
        pres[:, 0:-1] = pres_tmp

        dis = np.abs(pres[selected_ins, 0:-1] - pres[selected_ins, -1])
        selected_ins = data_ind[selected_ins]
        selected_lab = np.argmin(dis)

        return [(selected_ins, selected_lab)]


def seed_random_state(seed):
    """Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r can not be used to generate numpy.random.RandomState"
                     " instance" % seed)


class DummyClf():
    """This classifier handles training sets with only 0s or 1s to unify the
    interface.
    """

    def __init__(self):
        self.classes_ = [0, 1]

    def fit(self, X, y):
        self.cls = int(y[0])  # 1 or 0

    # def train(self, dataset):
    #     _, y = zip(*dataset.get_labeled_entries())
    #     self.cls = int(y[0])

    def predict(self, X):
        return self.cls * np.ones(len(X))

    def predict_real(self, X):
        return self.predict_proba(X) * 2 - 1

    def predict_proba(self, X):
        ret = np.zeros((len(X), 2))
        ret[:, self.cls] = 1.
        return ret


class _BinaryRelevance():
    r"""Binary Relevance

    base_clf : base-classifier
        If wanting to use predict_proba, base_clf are required to support
        predict_proba method.

    References
    ----------
    .. [1] Tsoumakas, Grigorios, Ioannis Katakis, and Ioannis Vlahavas. "Mining
           multi-label data." Data mining and knowledge discovery handbook.
           Springer US, 2009. 667-685.
    """

    def __init__(self, base_clf):
        self.base_clf = copy.copy(base_clf)
        self.clfs_ = None

    def train(self, X, Y):
        r"""Train model with given feature.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Train feature vector.

        y : array-like, shape=(n_samples, n_labels)
            Target labels.

        Attributes
        ----------
        clfs\_ : list of :py:mod:`libact.models` object instances
            Classifier instances.

        Returns
        -------
        self : object
            Retuen self.
        """
        # X = np.array(X)
        # Y = np.array(y)

        self.n_labels_ = np.shape(Y)[1]
        self.n_features_ = np.shape(X)[1]

        self.clfs_ = []
        for i in range(self.n_labels_):
            # TODO should we handle it here or we should handle it before calling
            if len(np.unique(Y[:, i])) == 1:
                clf = DummyClf()
            else:
                clf = copy.deepcopy(self.base_clf)
            clf.fit(X, Y[:, i])
            self.clfs_.append(clf)

        return self

    def predict(self, X):
        r"""Predict labels.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Feature vector.

        Returns
        -------
        pred : numpy array, shape=(n_samples, n_labels)
            Predicted labels of given feature vector.
        """
        X = np.asarray(X)
        if self.clfs_ is None:
            raise ValueError("Train before prediction")
        if X.shape[1] != self.n_features_:
            raise ValueError('Given feature size does not match')

        pred = np.zeros((X.shape[0], self.n_labels_))
        for i in range(self.n_labels_):
            pred[:, i] = self.clfs_[i].predict(X)
        return pred.astype(int)

    def predict_real(self, X):
        r"""Predict the probability of being 1 for each label.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Feature vector.

        Returns
        -------
        pred : numpy array, shape=(n_samples, n_labels)
            Predicted probability of each label.
        """
        X = np.asarray(X)
        if self.clfs_ is None:
            raise ValueError("Train before prediction")
        if X.shape[1] != self.n_features_:
            raise ValueError('given feature size does not match')

        pred = np.zeros((X.shape[0], self.n_labels_))
        for i in range(self.n_labels_):
            if isinstance(self.clfs_[i], DummyClf):
                pred[:, i] = self.clfs_[i].predict_real(X)[:, 1]
            elif isinstance(self.clfs_[i], SVC):
                value = self.clfs_[i].decision_function(X)
                if len(np.shape(value)) == 1:  # n_classes == 2
                    pred_value = np.vstack((-value, value)).T
                    pred[:, i] = pred_value[:, 1]
                else:
                    pred[:, i] = value[:, 1]
            elif isinstance(self.clfs_[i], LogisticRegression):
                value = self.clfs_[i].decision_function(X)
                if len(np.shape(value)) == 1:  # n_classes == 2
                    pred_value = np.vstack((-value, value)).T
                    pred[:, i] = pred_value[:, 1]
                else:
                    pred[:, i] = value[:, 1]
            # pred[:, i] = self.clfs_[i].predict_real(X)[:, 1]
        return pred

    def predict_proba(self, X):
        r"""Predict the probability of being 1 for each label.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Feature vector.

        Returns
        -------
        pred : numpy array, shape=(n_samples, n_labels)
            Predicted probability of each label.
        """
        X = np.asarray(X)
        if self.clfs_ is None:
            raise ValueError("Train before prediction")
        if X.shape[1] != self.n_features_:
            raise ValueError('given feature size does not match')

        pred = np.zeros((X.shape[0], self.n_labels_))
        for i in range(self.n_labels_):
            pred[:, i] = self.clfs_[i].predict_proba(X)[:, 1]
        return pred


class QueryMultiLabelMMC(BaseIndexQuery):
    """Maximum loss reduction with Maximal Confidence (MMC)
    This algorithm is designed to use binary relavance with SVM as base model.

    The implementation refers to the project: https://github.com/ntucllab/libact

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    base_learner :  object instance
        The base learner for binary relavance, should support predict_proba

    br_base : ProbabilisticModel object instance
        The base learner for the binary relevance in MMC.
        Should support predict_proba.

    logreg_param : dict, optional (default={})
        Setting the parameter for the logistic regression that are used to
        predict the number of labels for a given feature vector. Parameter
        detail please refer to:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    References
    ----------
    [1] Yang, Bishan, et al. "Effective multi-label active learning for text
        classification." Proceedings of the 15th ACM SIGKDD international
        conference on Knowledge discovery and data mining. ACM, 2009.
    """

    def __init__(self, X, y, *args, **kwargs):
        super(QueryMultiLabelMMC, self).__init__(X, y)
        self.n_samples, self.n_labels = np.shape(self.y)

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.logreg_param = kwargs.pop('logreg_param',
                                       {'multi_class': 'multinomial', 'solver': 'newton-cg',
                                        'random_state': random_state})
        self.logistic_regression_ = LogisticRegression(**self.logreg_param)

        self.br_base = kwargs.pop('br_base',
                                  SVC(kernel='linear', probability=True,
                                      random_state=random_state))
    
    def sequential_select(self, label_index, unlabel_index):
        """
            Select one unlabel-sample at a time.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.
        Returns
        -------
        selected_ind: int
            The selected index.
        """
        if isinstance(label_index, (list, np.ndarray)):
            label_index = IndexCollection(label_index)
        elif isinstance(label_index, MultiLabelIndexCollection):
            label_index = IndexCollection(label_index.get_unbroken_instances())
        elif not isinstance(label_index,IndexCollection):
            raise TypeError("index type error")
        if isinstance(unlabel_index, (list, np.ndarray)):
            unlabel_index = IndexCollection(unlabel_index)
        elif isinstance(unlabel_index, MultiLabelIndexCollection):
            unlabel_index = IndexCollection(unlabel_index.get_unbroken_instances())
        elif not isinstance(unlabel_index,IndexCollection):
            raise TypeError("index type error")
        
        if len(unlabel_index) <= 1:
            return list(unlabel_index)

        labeled_pool = self.X[label_index]
        X_pool = self.X[unlabel_index]

        br = _BinaryRelevance(self.br_base)
        br.train(self.X[label_index], self.y[label_index])

        trnf = br.predict_proba(labeled_pool)
        poolf = br.predict_proba(X_pool)
        f = poolf * 2 - 1

        trnf = np.sort(trnf, axis=1)[:, ::-1]
        trnf /= np.tile(trnf.sum(axis=1).reshape(-1, 1), (1, trnf.shape[1]))
        if len(np.unique(self.y.sum(axis=1))) == 1:
            lr = DummyClf()
        else:
            lr = self.logistic_regression_
        lr.fit(trnf, self.y[label_index].sum(axis=1))

        idx_poolf = np.argsort(poolf, axis=1)[:, ::-1]
        poolf = np.sort(poolf, axis=1)[:, ::-1]
        poolf /= np.tile(poolf.sum(axis=1).reshape(-1, 1), (1, poolf.shape[1]))
        pred_num_lbl = lr.predict(poolf).astype(int)

        yhat = -1 * np.ones((len(X_pool), self.n_labels), dtype=int)
        for i, p in enumerate(pred_num_lbl):
            yhat[i, idx_poolf[i, :p]] = 1

        score = ((1 - yhat * f) / 2).sum(axis=1)
        ask_id = self.random_state_.choice(np.where(score == np.max(score))[0])

        return unlabel_index[ask_id]
        
    def select(self, label_index, unlabel_index, batch_size=1):
        """
            Select the unlabel data in batch mode.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_ind: list
            The selected indexes.        
        """
        if isinstance(label_index, (list, np.ndarray)):
            label_index = IndexCollection(label_index)
        elif isinstance(label_index, MultiLabelIndexCollection):
            label_index = IndexCollection(label_index.get_unbroken_instances())
        elif not isinstance(label_index,IndexCollection):
            raise TypeError("index type error")
        if isinstance(unlabel_index, (list, np.ndarray)):
            unlabel_index = IndexCollection(unlabel_index)
        elif isinstance(unlabel_index, MultiLabelIndexCollection):
            unlabel_index = IndexCollection(unlabel_index.get_unbroken_instances())
        elif not isinstance(unlabel_index,IndexCollection):
            raise TypeError("index type error")
        
        if len(unlabel_index) <= batch_size:
            return list(unlabel_index)
        
        select_index = []
        for i in range(batch_size):
            selected = self.sequential_select(label_index, unlabel_index)
            label_index.update(selected)
            unlabel_index.difference_update(selected)
            select_index.append((selected, ))

        return select_index


class QueryMultiLabelAdaptive(BaseIndexQuery):
    r"""Adaptive Active Learning

    This approach combines Max Margin Uncertainty Sampling and Label
    Cardinality Inconsistency.

    The implementation refers to the project: https://github.com/ntucllab/libact

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
        
    base_clf : ContinuousModel object instance
        The base learner for binary relavance should support predict_proba() method.
        Such as sklearn.linear_model.LogisticRegression(solver='liblinear').

    betas : list of float, 0 <= beta <= 1, default: [0., 0.1, ..., 0.9, 1.]
        List of trade-off parameter that balances the relative importance
        degrees of the two measures.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    References
    ----------
    .. [1] Li, Xin, and Yuhong Guo. "Active Learning with Multi-Label SVM
           Classification." IJCAI. 2013.
    """

    def __init__(self, X, y, base_clf=LogisticRegression(solver='liblinear'), betas=None, random_state=None):
        super(QueryMultiLabelAdaptive, self).__init__(X, y)

        self.n_samples, self.n_labels = np.shape(self.y)

        self.base_clf = copy.deepcopy(base_clf)

        # TODO check beta value
        self.betas = betas
        if self.betas is None:
            self.betas = [i / 10. for i in range(0, 11)]

        self.random_state_ = seed_random_state(random_state)

    def sequential_select(self, label_index, unlabel_index):
        """
            Select one unlabel-data at a time.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.
        Returns
        -------
        selected_ind: int
            The selected index.
        """

        if isinstance(label_index, (list, np.ndarray)):
            label_index = IndexCollection(label_index)
        elif isinstance(label_index, MultiLabelIndexCollection):
            label_index = IndexCollection(label_index.get_unbroken_instances())
        elif not isinstance(label_index,IndexCollection):
            raise TypeError("index type error")
        if isinstance(unlabel_index, (list, np.ndarray)):
            unlabel_index = IndexCollection(unlabel_index)
        elif isinstance(unlabel_index, MultiLabelIndexCollection):
            unlabel_index = IndexCollection(unlabel_index.get_unbroken_instances())
        elif not isinstance(unlabel_index,IndexCollection):
            raise TypeError("index type error")
        
        if len(unlabel_index) <= 1:
            return list(unlabel_index)

        X_pool = self.X[unlabel_index]

        clf = _BinaryRelevance(self.base_clf)
        clf.train(self.X[label_index], self.y[label_index])
        real = clf.predict_real(X_pool)
        pred = clf.predict(X_pool)

        # Separation Margin
        pos = np.copy(real)
        pos[real <= 0] = np.inf
        neg = np.copy(real)
        neg[real >= 0] = -np.inf
        separation_margin = pos.min(axis=1) - neg.max(axis=1)
        uncertainty = 1. / separation_margin

        # Label Cardinality Inconsistency
        average_pos_lbl = self.y[label_index].mean(axis=0).sum()
        label_cardinality = np.sqrt((pred.sum(axis=1) - average_pos_lbl) ** 2)

        candidate_idx_set = set()
        for b in self.betas:
            # score shape = (len(X_pool), )
            score = uncertainty ** b * label_cardinality ** (1. - b)
            for idx in np.where(score == np.max(score))[0]:
                candidate_idx_set.add(idx)

        candidates = list(candidate_idx_set)

        approx_err = []
        for idx in candidates:
            br = _BinaryRelevance(self.base_clf)
            br.train(np.vstack((self.X[label_index], X_pool[idx])), np.vstack((self.y[label_index], pred[idx])))
            br_real = br.predict_real(X_pool)

            pos = np.copy(br_real)
            pos[br_real < 0] = 1
            pos = np.max((1. - pos), axis=1)

            neg = np.copy(br_real)
            neg[br_real > 0] = -1
            neg = np.max((1. + neg), axis=1)

            err = neg + pos

            approx_err.append(np.sum(err))

        choices = np.where(np.array(approx_err) == np.min(approx_err))[0]
        ask_idx = candidates[self.random_state_.choice(choices)]

        return unlabel_index[ask_idx]

    def select(self, label_index, unlabel_index, batch_size=1):
        """
            Select the unlabel data in batch mode.
        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_ind: list
            The selected indexes.
        """
        if isinstance(label_index, (list, np.ndarray)):
            label_index = IndexCollection(label_index)
        elif isinstance(label_index, MultiLabelIndexCollection):
            label_index = IndexCollection(label_index.get_unbroken_instances())
        elif not isinstance(label_index,IndexCollection):
            raise TypeError("index type error")
        if isinstance(unlabel_index, (list, np.ndarray)):
            unlabel_index = IndexCollection(unlabel_index)
        elif isinstance(unlabel_index, MultiLabelIndexCollection):
            unlabel_index = IndexCollection(unlabel_index.get_unbroken_instances())
        elif not isinstance(unlabel_index,IndexCollection):
            raise TypeError("index type error")
        
        if len(unlabel_index) <= batch_size:
            return list(unlabel_index)
        
        select_index = []
        for i in range(batch_size):
            selected = self.sequential_select(label_index, unlabel_index)
            label_index.update(selected)
            unlabel_index.difference_update(selected)
            select_index.append((selected, ))

        return select_index
        

class QueryMultiLabelRandom(BaseMultiLabelQuery):
    """Select instance or instance-label pairs randomly."""

    def select(self, label_index, unlabel_index, batch_size=1, select_type='ins-lab', **kwargs):
        """Select a subset from the unlabeled set, return the selected instance and label.

        Parameters
        ----------
        label_index: ignore

        unlabel_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        batch_size: int, optional (default=1)
            Selection batch size.

        select_type: {'ins', 'ins-lab'}
            The selection type.
            ins: select a batch of instances to query all of their labels.
            ins-lab: select a batch of instance-label pairs to query.

        Returns
        -------
        selected_ind: list
            The selected indexes. It is a list of tuples.
        """
        if select_type == 'ins':
            if len(unlabel_index) <= batch_size:
                return unlabel_index
            unkonwn_entries = self._check_multi_label_ind(unlabel_index)
            unkonwn_ins = unkonwn_entries.get_instance_index()
            perm = randperm(len(unkonwn_ins) - 1, batch_size)
            return [(unkonwn_ins[i],) for i in perm]
        elif select_type == 'ins-lab':
            if len(unlabel_index) <= batch_size:
                return unlabel_index
            unkonwn_entries = self._check_multi_label_ind(unlabel_index)
            perm = randperm(len(unkonwn_entries) - 1, batch_size)
            tpl = list(unkonwn_entries.index)
            return [tpl[i] for i in perm]
        else:
            raise ValueError("select_type must be one of {'ins', 'ins-lab'}")
