"""
Query strategies for feature querying setting.

We implement the following strategies:

1. KDD'18: Active Feature Acquisition with Supervised Matrix Completion (AFASMC).
2. ICDM'13: Active Matrix Completion using Query by Committee (QBC)
3. ICDM'13: Active Matrix Completion using Committee Stability (Stability)
4. Random
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
from scipy.sparse import csr_matrix

from ..index.index_collections import MultiLabelIndexCollection


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
        max_iter : The maximal iterations of optimization
    """
    max_iter = kwargs.pop('max_iter', 100)
    lambda1 = kwargs.pop('lambda1', 1)
    lambda2 = kwargs.pop('lambda2', 1)
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape
    lambda2 /= n_samples

    X_obr = np.zeros(n_samples * n_features)
    X_obr[omega] = X.flatten(order='F')[omega]
    X_obr = X_obr.reshape((n_features, n_samples)).T
    if isinstance(omega[0], tuple):
        omega = MultiLabelIndexCollection(omega, X.shape[1])
    else:
        omega = MultiLabelIndexCollection.construct_by_1d_array(omega, label_mat_shape=X.shape)
    obrT = np.asarray(omega.get_matrix_mask(label_mat_shape=X.shape, sparse_format='lil_matrix').todense())
    # X_obr = np.ma.array(data=X, mask=obrT.todense(), fill_value=0)
    # mask = (obrT == 1).todense()
    # X_obr[mask == 1] = X[mask == 1]

    theta0 = 1
    theta1 = 1
    Z0 = np.zeros((n_samples, n_features))
    Z1 = Z0
    ineqLtemp0 = np.zeros((n_samples, n_features))
    ineqLtemp1 = ineqLtemp0
    L = 2
    convergence = np.zeros((max_iter, 1))

    # train a linear model whose obj function is min{norm(Xw-Y)}
    X_extend = np.hstack((X_obr, np.ones((n_samples, 1))))
    W = np.linalg.pinv(X_extend.T.dot(X_extend)).dot(X_extend.T).dot(y)
    w = W[0:-1].flatten()
    b = W[-1]

    for k in range(max_iter):
        Y = Z1 + theta1 * (1 / theta0 - 1) * (Z1 - Z0)
        svd_obj_temp_temp = (theta1 * (1 / theta0 - 1) + 1) * ineqLtemp1 - theta1 * (
        1 / theta0 - 1) * ineqLtemp0 - X_obr
        svd_obj_temp = svd_obj_temp_temp + 2 * lambda2 * (Y.dot(w) + b - y).reshape(-1, 1).dot(w.reshape((1, -1)))
        svd_obj = Y - 1 / L * svd_obj_temp
        Z0 = Z1
        Z1, traceNorm = _svd_threshold(svd_obj, lambda1 / L)

        ineqLtemp0 = ineqLtemp1
        # do not know whether it is element wise or not
        ineqLtemp1 = Z1 * obrT
        ineqL = np.linalg.norm(ineqLtemp1 - X_obr, ord='fro') ** 2 / 2 + sum((Z1.dot(w) + b - y) ** 2) * lambda2

        ineqRtemp = sum(sum(svd_obj_temp_temp ** 2)) / 2 + sum((Y.dot(w) + b - y) ** 2) * lambda2 - svd_obj_temp.dot(
            Y.flatten())
        ineqR = ineqRtemp + svd_obj_temp.dot(Z1.flatten()) + L / 2 * sum(sum((Z1 - Y) ** 2))

        while ineqL > ineqR:
            L = L * 2
            svd_obj = Y - 1 / L * svd_obj_temp
            Z1, traceNorm = _svd_threshold(svd_obj, lambda1 / L)

            ineqLtemp1 = Z1 * obrT
            ineqL = np.linalg.norm(ineqLtemp1 - X_obr, ord='fro') ** 2 / 2 + sum((Z1.dot(w) + b - y) ** 2) * lambda2
            ineqR = ineqRtemp + svd_obj_temp.dot(Z1.flatten()) + L / 2 * sum(sum((Z1 - Y) ** 2))

        theta0 = theta1
        theta1 = (np.sqrt(theta1 ** 4 + 4 * theta1 ** 2) - theta1 ** 2) / 2

        convergence[k, 0] = ineqL + lambda1 * traceNorm

        # judge convergence
        if k == 0:
            minObj = convergence[k, 0]
            X_mc = Z1
        else:
            if convergence[k, 0] < minObj:
                minObj = convergence[k, 0]
                X_mc = Z1
        if k > 0:
            if np.abs(convergence[k, 0] - convergence[k - 1, 0]) < ((1e-6) * convergence[k - 1, 0]):
                break

    return X_mc, minObj, convergence
