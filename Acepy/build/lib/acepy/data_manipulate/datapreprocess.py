'''
Data Preprocess
'''
# Authors: Guo-Xiang Li
# License: BSD 3 clause

from __future__ import division
import numpy as np
from scipy import sparse

__all__ = [
    'minmax_scale',
    'StandardScale'
]


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

        
def minmax_scale(X, feature_range=(0, 1)):
    """Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data matrix with [n_samples, n_features].

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    
    Returns
    -------
    x_scaled : array-like, shape (n_samples, n_features)
        
    """

    if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
    
    if sparse.issparse(X):
            raise TypeError("MinMaxScaler does no support sparse input. "
                            "You may consider to use MaxAbsScaler instead.")
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
        
    data_min = np.nanmin(X, axis=0)
    data_max = np.nanmax(X, axis=0)
    data_dis = data_max - data_min
    data_dis = _handle_zeros_in_scale(data_dis)

    x_std = (X - data_min) / data_dis
    x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return x_scaled


def StandardScale(X):
    '''
        Standardize features by removing the mean and scaling to unit variance
    
    Parameters
    ----------
    X: array-like
        Data matrix with [n_samples, n_features]
    
    Returns
    -------
    x_StdS: array-like
        index of training set, shape like [n_split_count, n_training_indexes]


    '''
    if sparse.issparse(X):
            raise TypeError("MinMaxScaler does no support sparse input. "
                            "You may consider to use MaxAbsScaler instead.")
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if not(hasattr(X, 'mean') and hasattr(X, 'var')):
        raise TypeError("X array does not has mean or var methods")
    
    data_mean = X.mean(axis=0)
    data_var = X.var(axis=0)
    x_StdS = (X - data_mean) / data_var

    return x_StdS


if __name__ == '__main__':
    data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    data1 = [[-1, 2, 1], [-0.5, 6, 1], [0, 10, 1], [1, 18, 1]]
    print(StandardScale(data))
    print(minmax_scale(data1, (0, 1)))
