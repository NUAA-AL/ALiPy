"""
Query strategies for feature querying setting.

Feature missing is a serious problem in many applications, which may
lead to low quality of training data and further significantly
degrade the learning performance. While feature acquisition usually
involves special devices or complex processes, it is expensive to
acquire all feature values for the whole dataset. Thus, some works
try to solve this problem by active learning.

To support the feature querying setting, acepy implements the following strategies:

1. KDD'18: Active Feature Acquisition with Supervised Matrix Completion (AFASMC).
2. ICDM'13: Active Matrix Completion using Committee Stability (Stability)
3. Random
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_array

from .base import BaseFeatureQuery
from ..index.index_collections import MultiLabelIndexCollection
from ..utils.misc import randperm


def _svd_threshold(svd_obj, lambdadL):
    svd_obj = np.asarray(svd_obj)
    eig_obj = svd_obj.T.dot(svd_obj)
    eig_obj2 = eig_obj
    eig_tag = 0
    eig_obj_timer = 1e20
    while (eig_tag == 0) and (eig_obj_timer >= 1e8):
        try:
            D, V = np.linalg.eig(eig_obj2)
            sorted_args = np.argsort(D)
            V = V[:, sorted_args]
            D = D[sorted_args]
            eig_tag = 1
        except:
            eig_tag = 0
            eig_obj2 = round(eig_obj * eig_obj_timer) / eig_obj_timer
            eig_obj_timer = eig_obj_timer / 10

    # D = np.diag(D)
    D[D < 0] = 0
    D = np.sqrt(D)
    D2 = copy.copy(D)
    D2[D2 != 0] = 1.0 / D2[D2 != 0]
    D = np.diag(np.fmax(np.zeros(D.shape), D - lambdadL))
    Zk = svd_obj.dot((V.dot(np.diag(D2))).dot(D.dot(V.T)))
    traceNorm = np.sum(np.diag(D))

    return Zk, traceNorm


def AFASMC_mc(X, y, omega, **kwargs):
    """Perform matrix completion method in AFASMC.
    It will train a linear model to use the label information.

    Parameters
    ----------
    X: array
        The [n_samples, n_features] training matrix in which X_ij indicates j-th
        feature of i-th instance. The unobserved entries can be anything.

    y: array
        The The [n_samples] label vector.

    omega: {list, MultiLabelIndexCollection, numpy.ndarray}
        The indices for the observed entries of X.
        Should be a 1d array of indexes (in matlab way),
        or MultiLabelIndexCollection or a list
        of tuples with 2 elements, in which, the 1st element is the index of instance
        and the 2nd element is the index of label.

    kwargs:
        lambda1  : The lambda1 trade-off parameter
        lambda2  : The lambda2 trade-off parameter
        max_in_iter : The maximal iterations of inner optimization
        max_out_iter : The maximal iterations of the outer optimization

    Returns
    -------
    Xmc: array
        The completed matrix.

    References
    ----------
    [1] Active feature acquisition with supervised matrix completion.
        Sheng-Jun Huang, Miao Xu, Ming-Kun Xie, Masashi Sugiyama, Gang Niu and Songcan Chen
        In: Proceedings of the 24th ACM SIGKDD Conference on Knowledge Discovery and Data
        Mining (KDD'18), 2018.
    """
    n_samples, n_features = np.shape(X)

    # X_obr = np.zeros(n_samples * n_features)
    # X_obr[omega] = X.flatten(order='F')[omega]
    # X_obr = X_obr.reshape((n_features, n_samples)).T
    if not isinstance(omega, MultiLabelIndexCollection):
        try:
            if isinstance(omega[0], tuple):
                omega = MultiLabelIndexCollection(omega, n_features)
            else:
                omega = MultiLabelIndexCollection.construct_by_1d_array(omega, label_mat_shape=(n_samples, n_features))
        except:
            raise ValueError("Please pass a 1d array of indexes or MultiLabelIndexCollection "
                             "(column major, start from 0) or a list "
                             "of tuples with 2 elements, in which, the 1st element is the index of instance "
                             "and the 2nd element is the index of features.")

    obrT = omega.get_matrix_mask(mat_shape=(n_samples, n_features), sparse=False)

    return AFASMC_mask_mc(X=X, y=y, mask=obrT, check_para=False, **kwargs)


def AFASMC_mask_mc(X, y, mask, **kwargs):
    """Perform matrix completion method in AFASMC.
    It will train a linear model to use the label information.

    Parameters
    ----------
    X: array
        The [n_samples, n_features] training matrix in which X_ij indicates j-th
        feature of i-th instance. The unobserved entries can be anything.

    y: array
        The The [n_samples] label vector.

    mask: {list, np.ndarray}
        The mask matrix of known elements. the matrix should have the shape [n_samples, n_features].
        There must be only 1 and 0 in the matrix, in which, 1 means the corresponding element is known,
        otherwise, it will be cheated as an unknown element.

    kwargs:
        lambda1  : The lambda1 trade-off parameter
        lambda2  : The lambda2 trade-off parameter
        max_in_iter : The maximal iterations of inner optimization
        max_out_iter : The maximal iterations of the outer optimization

    Returns
    -------
    Xmc: array
        The completed matrix.

    References
    ----------
    [1] Active feature acquisition with supervised matrix completion.
        Sheng-Jun Huang, Miao Xu, Ming-Kun Xie, Masashi Sugiyama, Gang Niu and Songcan Chen
        In: Proceedings of the 24th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'18), 2018.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    max_in_iter = kwargs.pop('max_in_iter', 100)
    max_out_iter = kwargs.pop('max_out_iter', 10)
    lambda1 = kwargs.pop('lambda1', 1)
    lambda2 = kwargs.pop('lambda2', 1)
    check_para = kwargs.pop('check_para', True)
    obrT = mask
    if check_para:
        obrT = np.asarray(mask)
        assert obrT.shape[0] == X.shape[0] and obrT.shape[1] == X.shape[1]
        ue = np.unique(obrT)
        assert len(ue) == 2
        assert 0 in ue
        assert 1 in ue
    X_obr = np.zeros((n_samples, n_features))
    X_obr += obrT * X
    n_samples, n_features = X.shape
    lambda2 /= n_samples
    theta0 = 1
    theta1 = 1
    Z0 = np.zeros((n_samples, n_features))
    Z1 = np.zeros((n_samples, n_features))
    ineqLtemp0 = np.zeros((n_samples, n_features))
    ineqLtemp1 = ineqLtemp0
    L = 2
    converge_out = np.zeros((max_in_iter, 1))

    for i in range(max_out_iter):
        # train a linear model whose obj function is min{norm(Xw-Y)}
        X_extend = np.hstack((X_obr, np.ones((n_samples, 1))))
        W = np.linalg.pinv(X_extend.T.dot(X_extend)).dot(X_extend.T).dot(y)
        w = W[0:-1].flatten()
        b = W[-1]

        convergence = np.zeros(max_in_iter)

        for k in range(max_in_iter):
            Y = Z1 + theta1 * (1 / theta0 - 1) * (Z1 - Z0)
            svd_obj_temp_temp = (theta1 * (1 / theta0 - 1) + 1) * ineqLtemp1 - theta1 * (
                    1 / theta0 - 1) * ineqLtemp0 - X_obr
            svd_obj_temp = svd_obj_temp_temp + 2 * lambda2 * (Y.dot(w) + b - y).reshape(-1, 1).dot(w.reshape((1, -1)))
            svd_obj = Y - 1 / L * svd_obj_temp
            Z0 = copy.copy(Z1)
            Z1, traceNorm = _svd_threshold(svd_obj, lambda1 / L)

            ineqLtemp0 = ineqLtemp1
            # do not know whether it is element wise or not
            ineqLtemp1 = Z1 * obrT
            ineqL = np.linalg.norm(ineqLtemp1 - X_obr, ord='fro') ** 2 / 2 + sum((Z1.dot(w) + b - y) ** 2) * lambda2

            ineqRtemp = sum(sum(svd_obj_temp_temp ** 2)) / 2 + sum(
                (Y.dot(w) + b - y) ** 2) * lambda2 - svd_obj_temp.flatten().dot(
                Y.flatten())
            ineqR = ineqRtemp + svd_obj_temp.flatten().dot(Z1.flatten()) + L / 2 * sum(sum((Z1 - Y) ** 2))

            while ineqL > ineqR:
                L = L * 2
                svd_obj = Y - 1 / L * svd_obj_temp
                Z1, traceNorm = _svd_threshold(svd_obj, lambda1 / L)

                ineqLtemp1 = Z1 * obrT
                ineqL = np.linalg.norm(ineqLtemp1 - X_obr, ord='fro') ** 2 / 2 + sum((Z1.dot(w) + b - y) ** 2) * lambda2
                ineqR = ineqRtemp + svd_obj_temp.flatten().dot(Z1.flatten()) + L / 2 * sum(sum((Z1 - Y) ** 2))

            theta0 = theta1
            theta1 = (np.sqrt(theta1 ** 4 + 4 * theta1 ** 2) - theta1 ** 2) / 2

            convergence[k] = ineqL + lambda1 * traceNorm

            # judge convergence
            if k == 0:
                minObj = convergence[k]
                X_mc = Z1
            else:
                if convergence[k] < minObj:
                    minObj = convergence[k]
                    X_mc = Z1
            if k > 0:
                if np.abs(convergence[k] - convergence[k - 1]) < ((1e-6) * convergence[k - 1]):
                    break

        converge_out[i] = minObj
        if i == 0:
            minObj_out = converge_out[k]
            Xmc = X_mc
        else:
            if converge_out[k] < minObj_out:
                minObj_out = converge_out[k]
                Xmc = X_mc
        if i > 0:
            if abs(converge_out[k] - converge_out[k - 1]) < (1e-4 * converge_out[k - 1]):
                break

    return (1 - obrT) * Xmc + obrT * X


class IterativeSVD_mc:
    """The SVD method for matrix completion.
    The implementation is refered to "fancyimpute"
    whose github address is: https://github.com/iskandr/fancyimpute

    Parameters
    ----------
    rank: int, optional (default=10)
        The rank of the factorized matrix.

    convergence_threshold: float, optional (default=0.00001)
        The optimization precision.

    max_iters: int, optional (default=200)
        The max iteration of optimization.

    gradual_rank_increase: bool, optional (default=True)
        Whether to increase the value of rank gradually during the optimization procedure.

    svd_algorithm: string, default = "arpack"
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).

    min_value: float, optional (default=None)
        The min value constraints of the imputed value.

    max_value: float, optional (default=None)
        The max value constraints of the imputed value.
    """

    def __init__(
            self,
            rank=10,
            convergence_threshold=0.00001,
            max_iters=200,
            gradual_rank_increase=True,
            svd_algorithm="arpack",
            min_value=None,
            max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.rank = rank
        self.max_iters = max_iters
        self.svd_algorithm = svd_algorithm
        self.convergence_threshold = convergence_threshold
        self.gradual_rank_increase = gradual_rank_increase

    def _converged(self, X_old, X_new, missing_mask):
        F32PREC = np.finfo(np.float32).eps
        # check for convergence
        old_missing_values = X_old * missing_mask
        new_missing_values = X_new * missing_mask
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        # edge cases
        if old_norm_squared == 0 or \
                (old_norm_squared < F32PREC and ssd > F32PREC):
            return False
        else:
            return (ssd / old_norm_squared) < self.convergence_threshold

    def _clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def impute(self, X, observed_mask, check_para=True):
        """Complete the matrix.

        Parameters
        ----------
        X: array
            The complete data matrix. Shape like [n_samples, n_features]

        observed_mask: {list, np.ndarray}
            The mask matrix of X. the matrix should have the same shape with X.
            There must be only 1 and 0 in the matrix, in which, 1 means the corresponding element is known,
            otherwise, it will be cheated as an unknown element.

        Returns
        -------
        X_filled: array
            The completed matrix.
        """
        X = check_array(X, force_all_finite=False)
        observed_mask = np.asarray(observed_mask)
        if check_para:
            obrT = np.asarray(observed_mask)
            assert obrT.shape[0] == X.shape[0] and obrT.shape[1] == X.shape[1]
            ue = np.unique(obrT)
            assert len(ue) == 2
            assert 0 in ue
            assert 1 in ue
        missing_mask = 1 - observed_mask
        X_filled = X * missing_mask
        for i in range(self.max_iters):
            # deviation from original svdImpute algorithm:
            # gradually increase the rank of our approximation
            if self.gradual_rank_increase:
                curr_rank = min(2 ** i, self.rank)
            else:
                curr_rank = self.rank
            tsvd = TruncatedSVD(curr_rank, algorithm=self.svd_algorithm)
            X_reduced = tsvd.fit_transform(X_filled)
            X_reconstructed = tsvd.inverse_transform(X_reduced)
            X_reconstructed = self._clip(X_reconstructed)

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)
            X_filled = X_reconstructed * missing_mask + X * observed_mask
            if converged:
                break
        return X_filled


class QueryFeatureAFASMC(BaseFeatureQuery):
    """This class implement the KDD'18: Active Feature Acquisition with
    Supervised Matrix Completion (AFASMC) method. It will complete the
    matrix with supervised information first. And select the missing feature
    with the highest variance based on the results of previous completion.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    train_idx: array-like
        the index of training data.

    References
    ----------
    [1] Active feature acquisition with supervised matrix completion.
        Sheng-Jun Huang, Miao Xu, Ming-Kun Xie, Masashi Sugiyama, Gang Niu
        and Songcan Chen In: Proceedings of the 24th ACM SIGKDD Conference
        on Knowledge Discovery and Data Mining (KDD'18), 2018.

    """

    def __init__(self, X, y, train_idx=None):
        super(QueryFeatureAFASMC, self).__init__(X, y)
        self._train_idx = train_idx
        self._feature_variance = []

    def select(self, observed_entries, unkonwn_entries, **kwargs):
        """Select a subset from the unlabeled set, return the selected instance and feature.

        Parameters
        ----------
        observed_entries: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0)
            or MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of features.

        unkonwn_entries: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0)
            or MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of features.

        Returns
        -------
        selected_feature: list
            The selected features, it is a list of tuples.
            Note that, the index is for the the WHOLE dataset, NOT THE TRAINING SET.
        """
        # build map from value to index
        if len(unkonwn_entries) <= 1:
            return unkonwn_entries
        unkonwn_entries = self._check_feature_ind(unkonwn_entries)
        observed_entries = self._check_feature_ind(observed_entries)

        if self._train_idx is None:
            obi = observed_entries.get_instance_index()
            uni = unkonwn_entries.get_instance_index()
            self._train_idx = np.union1d(obi, uni)

        # map entries of the whole data to the training data
        tr_ob = []
        for entry in observed_entries:
            # if entry[0] in self._train_idx:
            ind_in_train = np.argwhere(self._train_idx == entry[0])[0][0]
            tr_ob.append((ind_in_train, entry[1]))
            # else:
            #     tr_ob.append(entry)
        tr_ob = MultiLabelIndexCollection(tr_ob)

        # matrix completion
        X_mc = AFASMC_mc(X=self.X[self._train_idx], y=self.y[self._train_idx], omega=tr_ob, **kwargs)
        self._feature_variance.append(X_mc)
        mc_sh = np.shape(X_mc)
        if len(self._feature_variance) >= 2:
            var_mat = np.zeros(mc_sh)
            for i in range(mc_sh[0]):
                for j in range(mc_sh[1]):
                    var_mat[i, j] = np.var([mat[i][j] for mat in self._feature_variance])
            var_mat *= 1 - tr_ob.get_matrix_mask(mat_shape=mc_sh, sparse=False)
            # in the first 2 rounds, all entries are 0, you will select the known elements.
            selected_feature = np.argmax(var_mat)  # a 1d index in training set
        else:
            selected_feature = np.argmax(1 - tr_ob.get_matrix_mask(mat_shape=mc_sh, sparse=False))

        return [(self._train_idx[selected_feature // mc_sh[1]], selected_feature % mc_sh[1])]

    def select_by_mask(self, observed_mask, **kwargs):
        """Select a subset from the unlabeled set by providing the mask matrix, 
        return the selected instance and feature.

        Parameters
        ----------
        observed_mask: {list, np.ndarray}
            The mask matrix of training set. the matrix should have the shape [n_train_idx, n_features].
            There must be only 1 and 0 in the matrix, in which, 1 means the corresponding element is known,
            otherwise, it will be cheated as an unknown element.

        Returns
        -------
        selected_feature: list
            The selected features, it is a list of tuples.
            Note that, the index is for the given mask, NOT the whole dataset.
        """
        if self._train_idx is None:
            raise ValueError("Please pass the indexes of training data when initializing.")
        observed_mask = np.asarray(observed_mask)
        assert len(observed_mask.shape) == 2
        assert observed_mask.shape[0] == len(self._train_idx)
        assert observed_mask.shape[1] == self.X.shape[1]

        X_mc = AFASMC_mask_mc(X=self.X[self._train_idx], y=self.y[self._train_idx], mask=observed_mask,
                              check_para=False, **kwargs)
        self._feature_variance.append(X_mc)
        mc_sh = np.shape(X_mc)
        if len(self._feature_variance) >= 2:
            var_mat = np.zeros(mc_sh)
            for i in range(mc_sh[0]):
                for j in range(mc_sh[1]):
                    var_mat[i, j] = np.var([mat[i][j] for mat in self._feature_variance])
            var_mat *= 1 - observed_mask
            # in the first 2 rounds, all entries are 0, you will select the known elements.
            selected_feature = np.argmax(var_mat)  # a 1d index in training set
        else:
            selected_feature = np.argmax(1 - observed_mask)

        return [(selected_feature // mc_sh[1], selected_feature % mc_sh[1])]


class QueryFeatureRandom(BaseFeatureQuery):
    """Randomly pick a missing feature to query."""

    def select(self, observed_entries, unkonwn_entries, batch_size=1, **kwargs):
        """Select a subset from the unlabeled set randomly.

        Parameters
        ----------
        observed_entries: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0)
            or MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of features.

        unkonwn_entries: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0)
            or MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of features.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_feature: list
            The selected features, it is a list of tuples.
            Note that, the index is a subset of unkonwn_entries.
        """
        # build map from value to index
        if len(unkonwn_entries) <= 1:
            return unkonwn_entries
        unkonwn_entries = self._check_feature_ind(unkonwn_entries)
        perm = randperm(len(unkonwn_entries) - 1, batch_size)
        tpl = list(unkonwn_entries.index)
        return [tpl[i] for i in perm]


class QueryFeatureStability(BaseFeatureQuery):
    """This class implement the Active Matrix Completion using Committee Stability method
    in ICDM'13: Active Matrix Completion.
    This method use different rank values in SVD matrix completion to construct committee.
    The uncertainty of prediction of each missing entry was computed as the variance of
    the values from the committee members for that entry.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    train_idx: array-like
        the index of training data.

    committee_rank: list, optional (default=None)
        The rank parameters used to contruct committee.
        Note that, each rank should be lower then min(X.shape)

    References
    ----------
    [1] Shayok Chakraborty, Jiayu Zhou, Vineeth Balasubramanian,
        Sethuraman Panchanathan, Ian Davidson, and Jieping Ye. 2013.
        Active matrix completion. In IEEE International Conference
        on Data Mining. 81-90.

    """

    def __init__(self, X, y, train_idx=None, rank_arr=None):
        super(QueryFeatureStability, self).__init__(X, y)
        self._train_idx = train_idx
        if rank_arr is None:
            rank_arr = [1, 2, 3]
        self._committee = []
        for rk in rank_arr:
            self._committee.append(IterativeSVD_mc(rank=rk))

    def select(self, observed_entries, unkonwn_entries, **kwargs):
        """Select a subset from the unlabeled set, return the selected instance and feature.

        Parameters
        ----------
        observed_entries: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0)
            or MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of features.

        unkonwn_entries: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0)
            or MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of features.

        Returns
        -------
        selected_feature: list
            The selected features, it is a list of tuples.
            Note that, the index is for the the WHOLE dataset, NOT THE TRAINING SET.
        """
        # build map from value to index
        if len(unkonwn_entries) <= 1:
            return unkonwn_entries
        unkonwn_entries = self._check_feature_ind(unkonwn_entries)
        observed_entries = self._check_feature_ind(observed_entries)

        if self._train_idx is None:
            obi = observed_entries.get_instance_index()
            uni = unkonwn_entries.get_instance_index()
            self._train_idx = np.union1d(obi, uni)

        # map entries of the whole data to the training data
        tr_ob = []
        for entry in observed_entries:
            # if entry[0] in self._train_idx:
            ind_in_train = np.argwhere(self._train_idx == entry[0])[0][0]
            tr_ob.append((ind_in_train, entry[1]))
            # else:
            #     tr_ob.append(entry)
        tr_ob = MultiLabelIndexCollection(tr_ob)

        selected_feature_in_train = self.select_by_mask(
            observed_mask=tr_ob.get_matrix_mask(mat_shape=(len(self._train_idx), self.X.shape[1]), sparse=False))

        return [(self._train_idx[selected_feature_in_train[0][0]], selected_feature_in_train[0][1])]

    def select_by_mask(self, observed_mask, **kwargs):
        """Select a subset from the unlabeled set by providing the mask matrix,
        return the selected instance and feature.

        Parameters
        ----------
        observed_mask: {list, np.ndarray}
            The mask matrix of training set. the matrix should have the shape [n_train_idx, n_features].
            There must be only 1 and 0 in the matrix, in which, 1 means the corresponding element is known,
            otherwise, it will be cheated as an unknown element.

        Returns
        -------
        selected_feature: list
            The selected features, it is a list of tuples.
            Note that, the index is for the given mask, NOT the whole dataset.
        """
        if self._train_idx is None:
            raise ValueError("Please pass the indexes of training data when initializing.")
        observed_mask = np.asarray(observed_mask)
        assert len(observed_mask.shape) == 2
        assert observed_mask.shape[0] == len(self._train_idx)
        assert observed_mask.shape[1] == self.X.shape[1]

        X_mc_arr = []
        for comt in self._committee:
            X_mc_arr.append(comt.impute(X=self.X[self._train_idx], observed_mask=observed_mask))

        mc_sh = observed_mask.shape
        if len(X_mc_arr) >= 2:
            var_mat = np.zeros(mc_sh)
            for i in range(mc_sh[0]):
                for j in range(mc_sh[1]):
                    var_mat[i, j] = np.var([mat[i][j] for mat in X_mc_arr])
            var_mat *= 1 - observed_mask
            # in the first 2 rounds, all entries are 0, you will select the known elements.
            selected_feature = np.argmax(var_mat)  # a 1d index in training set
        else:
            selected_feature = np.argmax(1 - observed_mask)

        return [(selected_feature // mc_sh[1], selected_feature % mc_sh[1])]
