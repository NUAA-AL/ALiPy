from abc import abstractmethod, ABCMeta

import numpy as np
from sklearn.utils.validation import check_X_y

from acepy.index import MultiLabelIndexCollection
from acepy.oracle import Oracle, Oracles
from acepy.utils.interface import BaseQueryStrategy


class BaseIndexQuery(BaseQueryStrategy, metaclass=ABCMeta):
    """The base class for the selection method which imposes a constraint on the parameters of select()"""

    @abstractmethod
    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Select instances to query.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.
        """


class BaseNoisyOracleQuery(BaseQueryStrategy, metaclass=ABCMeta):
    def __init__(self, X, y, oracles):
        super(BaseNoisyOracleQuery, self).__init__(X, y)
        if isinstance(oracles, list):
            self._oracles_type = 'list'
            for oracle in oracles:
                assert isinstance(oracle, Oracle)
        elif isinstance(oracles, Oracles):
            self._oracles_type = 'Oracles'
        else:
            raise TypeError("The type of parameter oracles must be a list or acepy.oracle.Oracles object.")
        self._oracles = oracles
        self._oracles_iterset = list(range(len(oracles))) if self._oracles_type == 'list' else oracles.names()
        self._oracle_ind_name_dict = dict(enumerate(self._oracles_iterset))

    @abstractmethod
    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Query from oracles. Return the selected instance and oracle.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.
        """


class BaseMultiLabelQuery(BaseIndexQuery, metaclass=ABCMeta):
    """Base query strategy for multi label setting."""

    def _check_multi_label_ind(self, container):
        """Check if the given array is an array of multi label indexes."""
        if not isinstance(container, MultiLabelIndexCollection):
            try:
                if isinstance(container[0], tuple):
                    container = MultiLabelIndexCollection(container, self.y.shape[1])
                else:
                    container = MultiLabelIndexCollection.construct_by_1d_array(container, label_mat_shape=self.y.shape)
            except:
                raise ValueError(
                    "Please pass a 1d array of indexes or MultiLabelIndexCollection (column major, "
                    "start from 0) or a list "
                    "of tuples with 2 elements, in which, the 1st element is the index of instance "
                    "and the 2nd element is the index of label.")
        return container

    def _check_multi_label(self, matrix):
        """Check if the given matrix is multi label"""
        # ytype = type_of_target(matrix)
        # if 'multilabel' not in ytype:
        if len(np.shape(matrix)) != 2:
            raise ValueError("Please provide a multi-label matrix in y with the shape [n_samples, n_classes].")

    def __init__(self, X=None, y=None, **kwargs):
        if X is not None and y is not None:
            self._check_multi_label(y)
            if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                # will not use additional memory
                check_X_y(X, y, accept_sparse='csc', multi_output=True)
                self.X = X
                self.y = y
            else:
                self.X, self.y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        else:
            self.X = X
            self.y = y

    @abstractmethod
    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
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

        batch_size: int, optional (default=1)
            Selection batch size.
        """


class BaseFeatureQuery(BaseMultiLabelQuery, metaclass=ABCMeta):
    """Base query strategy for feature querying setting.
    Basically have the same api with multi label setting."""

    def _check_feature_ind(self, container):
        if not isinstance(container, MultiLabelIndexCollection):
            try:
                if isinstance(container[0], tuple):
                    container = MultiLabelIndexCollection(container, self.X.shape[1])
                else:
                    container = MultiLabelIndexCollection.construct_by_1d_array(container, label_mat_shape=self.X.shape)
            except:
                raise ValueError(
                    "Please pass a 1d array of indexes or MultiLabelIndexCollection (column major, start from 0)"
                    "or a list of tuples with 2 elements, in which, the 1st element is the index of instance "
                    "and the 2nd element is the index of features.")
        return container

    def __init__(self, X=None, y=None, **kwargs):
        if X is not None and y is not None:
            if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                # will not use additional memory
                check_X_y(X, y, accept_sparse='csc', multi_output=True)
                self.X = X
                self.y = y
            else:
                self.X, self.y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        else:
            self.X = X
            self.y = y

    @abstractmethod
    def select(self, observed_entries, unkonwn_entries, batch_size=1, **kwargs):
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

        batch_size: int, optional (default=1)
            Selection batch size.
        """
