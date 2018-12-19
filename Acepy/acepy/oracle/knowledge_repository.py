"""
Knowledge repository
Store the information given by the oracle (labels, cost, etc.).

Functions include:
1. Retrieving
2. History recording
3. Get labeled set for training model

Nomally, we only store the indexes during the active learning
process for memory efficiency. However, we also provide
this class to store the queried instances and labels for additional usage.
This is a container to store and retrieve queries in detail.
It provide the function to return the labeled matrix X,y
for training and querying history for auditing.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import copy
import numpy as np
import prettytable as pt
from sklearn.utils.validation import check_array

from ..utils.ace_warnings import *
from ..utils.interface import BaseRepository
from ..utils.misc import _is_arraylike, check_one_to_one_correspondence
from ..index.index_collections import IndexCollection, MultiLabelIndexCollection
from ..utils.misc import unpack


class ElementRepository(BaseRepository):
    """Class to store fine-grained (element-wise) data.

    Both the example and label are not required to be an array-like object,
    they can be complicated object.

    Parameters
    ----------
    labels:  {list, numpy.ndarray}
        Labels of initial labeled set. shape [n_samples]

    indexes: {list, numpy.ndarray, IndexCollection}
        Indexes of labels, should have the same length and is one-to-one correspondence of labels.

    examples: array-like, optional (default=None)
        Array of examples, should have the same length and is one-to-one correspondence of labels.

    Examples
    ----------
    >>> import numpy as np
    >>> X = np.random.randn(100, 20)    # 100 instances in total
    >>> y = np.random.randn(100)
    >>> label_ind = [11,32,0,6,74]

    >>> repo = ElementRepository(labels=y[label_ind], indexes=label_ind, examples=X[label_ind])
    >>> # asume you have queried once and got some information
    >>> select_ind = [12]
    >>> repo.update_query(labels=y[select_ind], indexes=select_ind, examples=X[select_ind])
    >>> X_train, y_train, ind_train = repo.get_training_data()
    """

    def __init__(self, labels, indexes, examples=None):
        """initialize supervised information.
        """
        # check and record parameters
        assert isinstance(labels, (list, np.ndarray))
        assert isinstance(indexes, (list, np.ndarray, IndexCollection, MultiLabelIndexCollection))
        if not check_one_to_one_correspondence(labels, indexes, examples):
            raise ValueError("Different length of given indexes and labels found.")

        # self._y = copy.copy(_labels)
        self._index_len = len(labels)
        self._indexes = list(indexes)
        self._instance_flag = False if examples is None else True

        # several indexes construct
        if self._instance_flag:
            examples = [tuple(vec) for vec in examples]
            self._exa2ind = dict(zip(examples, self._indexes))
            self._ind2exa = dict(zip(self._indexes, examples))
        self._ind2label = dict(zip(self._indexes, labels))

        # record
        self.cost_inall = 0
        self._cost_arr = []
        self.num_of_queries = 0
        self._query_history = []

    def __contains__(self, item):
        return item in self._ind2label.keys()

    def add(self, select_index, label, cost=None, example=None):
        """Add an element to the repository.

        Parameters
        ----------
        select_index: int or tuple
            The selected index in active learning.

        label: object
            Supervised information given by the oracle.

        cost: object, optional (default=None)
            Cost produced by querying, given by the oracle.

        example: object, optional (default=None)
            Instance for adding.
        """
        if self._instance_flag:
            if example is None:
                raise Exception("This repository has the instance information,"
                                "must provide example parameter when adding entry")
            example = tuple(example)
            self._exa2ind[example] = select_index
            self._ind2exa[select_index] = example

        if cost is not None:
            self.cost_inall += np.sum(cost)
            self._cost_arr.append(cost)

        self._ind2label[select_index] = label
        self._indexes.append(select_index)
        return self

    def discard(self, index=None, example=None):
        """Discard element either by index or example.

        Must provide one of them.

        Parameters
        ----------
        index: int or tuple, optional (default=None)
            Index to discard.

        example: object, optional (default=None)
            Example to discard, must be one of the instance in data repository.
        """
        if index is None and example is None:
            raise Exception("Must provide one of index or example.")
        if index is not None:
            if index not in self._indexes:
                warnings.warn("Index %s is not in the repository, skipped." % str(index),
                              category=ValidityWarning,
                              stacklevel=3)
                return self
            self._indexes.remove(index)
            self._ind2label.pop(index)
            if self._instance_flag:
                self._exa2ind.pop(self._ind2exa.pop(index))

        if example is not None:
            if not self._instance_flag:
                raise Exception("This data base is not initialized with examples, discard by example is illegal.")
            else:
                example = tuple(example)
                if example not in self._exa2ind:
                    warnings.warn("example %s is not in the repository, skipped." % str(example),
                                  category=ValidityWarning,
                                  stacklevel=3)
                    return self
                ind = self._exa2ind[example]
                self._indexes.remove(ind)
                self._exa2ind.pop(example)
                self._ind2exa.pop(ind)
                self._ind2label.pop(ind)
        return self

    def update_query(self, indexes, labels, cost=None, examples=None):
        """Updating data base with queried information.

        The elements in the parameters should be one-to-one correspondence.

        Parameters
        ----------
        indexes: {list, numpy.ndarray, IndexCollection}
            Indexes of selected instances.

        labels: {list, numpy.ndarray}
            Labels to be updated.

        cost: array-like or object, optional (default=None)
            cost corresponds to the query.

        examples: array-like or object, optional (default=None)
            examples to be updated.
        """
        if not check_one_to_one_correspondence(labels, indexes, cost, examples):
            raise ValueError("Different length of parameters found. "
                             "They should have the same length and is one-to-one correspondence.")
        labels, indexes, cost, examples = unpack(labels, indexes, cost, examples)
        if not isinstance(indexes, (list, np.ndarray)):
            self.add(label=labels, select_index=indexes, cost=cost, example=examples)
        else:
            for i in range(len(labels)):
                self.add(label=labels[i], select_index=indexes[i], example=examples[i] if examples is not None else None,
                         cost=cost[i] if cost is not None else None)
        self.num_of_queries += 1
        self._update_query_history(labels, indexes, cost)
        return self

    def retrieve_by_indexes(self, indexes):
        """Retrieve by indexes.

        Parameters
        ----------
        indexes: array-like or object
            The indexes used for retrieving.
            Note that, if you want to retrieve by 2 or more indexes, a list or np.ndarray is expected.
            Otherwise, it will be treated as only one index.

        Returns
        -------
        X: array-like
            The retrieved instances.

        y: array-like
            The retrieved labels.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            indexes = [indexes]
        example_arr = []
        label_arr = []
        for k in indexes:
            if k in self._ind2label.keys():
                label_arr.append(self._ind2label[k])
                if self._instance_flag:
                    example_arr.append(self._ind2exa[k])
            else:
                warnings.warn("Index %s for retrieving is not in the repository, skip." % str(k),
                              category=ValidityWarning)
        return np.asarray(example_arr), np.asarray(label_arr)

    def retrieve_by_examples(self, examples):
        """Retrieve by examples.

        Parameters
        ----------
        examples: array-like or object
            The examples used for retrieving. Should be a subset in the repository.

        Returns
        -------
        y: array-like
            The retrieved labels.
        """
        if not self._instance_flag:
            raise Exception("This repository do not have the instance information, "
                            "retrieve_by_examples is not supported")
        if not isinstance(examples, (list, np.ndarray)):
            examples = [examples]
        elif len(np.shape(examples)) == 1:
            examples = [examples]
        q_id = []
        for k in examples:
            k = tuple(k)
            if k in self._exa2ind.keys():
                q_id.append(self._exa2ind[k])
            else:
                warnings.warn("Example for retrieving is not in the repository, skip.",
                              category=ValidityWarning)
        return self.retrieve_by_indexes(q_id)

    def get_training_data(self):
        """Get training set.

        Returns
        -------
        X_train: list, shape (n_training_examples, n_features)
            The feature matrix of training data.

        y_train: list
            The labels of training data.

        indexes: list
            The indexes of the instances and labels.
            e.g. the first row of X_train is the indexes[0] instance in feature matrix X.
        """
        X_train = []
        y_train = []
        for ind in self._indexes:
            if self._instance_flag:
                X_train.append(self._ind2exa[ind])
            y_train.append(self._ind2label[ind])
        return np.asarray(X_train), np.asarray(y_train), np.asarray(self._indexes)

    def clear(self):
        """Clear this container."""
        self._indexes.clear()
        self._exa2ind.clear()
        self._ind2label.clear()
        self._indexes.clear()
        self._instance_flag = False
        self.cost_inall = 0
        self._cost_arr = []
        self.num_of_queries = 0
        self._query_history.clear()

    def _update_query_history(self, labels, indexes, cost):
        """record the query history"""
        self._query_history.append(((labels, cost), indexes))

    def full_history(self):
        """Return full version of query history"""
        tb = pt.PrettyTable()
        # tb.set_style(pt.MSWORD_FRIENDLY)
        for query_ind in range(len(self._query_history)):
            query_result = self._query_history[query_ind]
            tb.add_column(str(query_ind), ["query_index:%s\nresponse:%s\ncost:%s" % (
                          str(query_result[1]), str(query_result[0][0]), str(query_result[0][1]))])
        tb.add_column('in all', ["number_of_queries:%s\ncost:%s" % (str(len(self._query_history)), str(self.cost_inall))])
        return str(tb)


# this class can only deal with the query-all-labels setting
class MatrixRepository(BaseRepository):
    """Matrix Knowledge Repository.

    The element of examples should be a vector to construct a matrix.
    Element of labels can be a single value or a vector for multi-label.

    Parameters
    ----------
    labels:  {list, numpy.ndarray}
        Labels of initial labeled set. shape [n_samples] or [n_samples, n_classes]

    indexes: {list, numpy.ndarray, IndexCollection}
        Indexes of labels, should have the same length with labels.

    examples: array-like, optional (default=None)
        Array of examples, shape [n_samples, n_features].

    Examples
    --------
    """

    def __init__(self, labels, indexes, examples=None):
        # check and record parameters
        if not check_one_to_one_correspondence(labels, indexes, examples):
            raise ValueError("Different length of the given parameters found.")

        self._y = check_array(labels, ensure_2d=False, dtype=None)
        self._indexes = np.asarray(indexes)
        self._instance_flag = False if examples is None else True
        if self._instance_flag:
            self._X = check_array(examples, accept_sparse='csr', ensure_2d=True, order='C')

        # record
        self.cost_inall = 0
        self._cost_arr = []
        self.num_of_queries = 0
        self._query_history = []

    def __contains__(self, item):
        return item in self._indexes

    def add(self, label, select_index, cost=None, example=None):
        """Add an element to the repository.

        Parameters
        ----------
        select_index: int or tuple
            The selected index in active learning.

        label: object
            Label given by oracle, shape should be the same with the labels in initializing.

        cost: object
            Cost produced by query, given by the oracle.

        example: array
            Feature vector of an instance.
        """
        # check validation
        if select_index in self._indexes:
            warnings.warn("Repeated index is found when adding element to knowledge repository. Skip this item",
                          category=RepeatElementWarning)
            return self
        if self._y.ndim == 1:
            if hasattr(label, '__len__'):
                raise TypeError("The initialized label array only have 1 dimension, "
                                "but received an array like label: %s." % str(label))
            self._y = np.append(self._y, [label])
        else:
            # this operation will check the validity automatically.
            self._y = np.append(self._y, [label], axis=0)
        self._indexes = np.append(self._indexes, select_index)
        self._cost_arr.append(cost)
        self.cost_inall += np.sum(cost) if cost is not None else 0
        if self._instance_flag:
            if example is None:
                raise Exception("This repository has the instance information,"
                                "must provide example parameter when adding entry")
            else:
                self._X = np.append(self._X, [example], axis=0)
        return self

    def update_query(self, labels, indexes, cost=None, examples=None):
        """Updating repository with queried information.

        The elements in the parameters should be one-to-one correspondence.

        Parameters
        ----------
        labels: {list, numpy.ndarray}
            Labels to be updated.

        indexes: {list, numpy.ndarray, IndexCollection}
            Indexes of selected instances.

        cost: array-like or object, optional (default=None)
            cost corresponds to the query.

        examples: array-like or object, optional (default=None)
            examples to be updated.
        """
        if not check_one_to_one_correspondence(labels, indexes, cost, examples):
            raise ValueError("Different length of parameters found. "
                             "They should have the same length and is one-to-one correspondence.")
        labels, indexes, cost, examples = unpack(labels, indexes, cost, examples)
        if not isinstance(indexes, (list, np.ndarray)):
            self.add(label=labels, select_index=indexes, cost=cost, example=examples)
        else:
            for i in range(len(indexes)):
                self.add(label=labels[i], select_index=indexes[i], cost=cost[i] if cost is not None else None,
                         example=examples[i] if examples is not None else None)
        self.num_of_queries += 1
        self._update_query_history(labels, indexes, cost)
        return self

    def discard(self, index=None, example=None):
        """Discard element either by index or example.

        Must provide one of them.

        Parameters
        ----------
        index: int, optional (default=None)
            Index to discard.

        example: object, optional (default=None)
            Example to discard, must be one of the instance in data repository.
        """
        if index is None and example is None:
            raise Exception("Must provide one of index or example")

        if index is not None:
            if index not in self._indexes:
                warnings.warn("Index %s is not in the repository, skipped." % str(index),
                              category=ValidityWarning,
                              stacklevel=3)
                return self
            ind = np.argwhere(self._indexes == index)

        if example is not None:
            if not self._instance_flag:
                raise Exception("This repository is not initialized with examples, discard by example is illegal.")
            else:
                ind = self._find_one_example(example)
            if ind == -1:
                warnings.warn("Example %s for retrieving is not in the repository, skip." % str(example),
                              category=ValidityWarning)
                return self

        mask = np.ones(len(self._indexes), dtype=bool)
        mask[ind] = False
        self._y = self._y[mask]
        self._indexes = self._indexes[mask]
        if self._instance_flag:
            self._X = self._X[mask]
        return self

    def retrieve_by_indexes(self, indexes):
        """Retrieve by indexes.

        Parameters
        ----------
        indexes: {list, numpy.ndarray}
            The indexes used for retrieving.
            Note that, if you want to retrieve by 2 or more indexes, a list of int is expected.

        Returns
        -------
        X: array-like
            The retrieved instances.

        y: array-like
            The retrieved labels.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            ind = np.argwhere(self._indexes == indexes)  # will return empty array if not found.
            if not ind:
                warnings.warn("Index %s for retrieving is not in the repository, skip." % str(indexes),
                              category=ValidityWarning)
            return self._X[ind,] if self._instance_flag else None, self._y[ind,]
        else:
            ind = [np.argwhere(self._indexes == indexes[i]).flatten()[0] for i in range(len(indexes))]
            return self._X[ind,] if self._instance_flag else None, self._y[ind,]

    def _find_one_example(self, example):
        """Find the index of the given example in the repository."""
        found = False
        for id, vec in enumerate(self._X):
            if np.all(vec == example):
                ind = id
                found = True
                break
        if not found:
            return -1
        return ind

    def retrieve_by_examples(self, examples):
        """Retrieve by examples.

        Parameters
        ----------
        examples: array-like or object
            The examples used for retrieving. Should be a subset in the repository.

        Returns
        -------
        y: array-like
            The retrieved labels.
        """
        if not self._instance_flag:
            raise Exception("This data base is not initialized with _examples, retrieve by example is illegal.")
        examples = np.asarray(examples)
        if examples.ndim == 1:
            ind = self._find_one_example(examples)
            if ind == -1:
                warnings.warn("Example %s for retrieving is not in the repository, skip." % str(examples),
                              category=ValidityWarning)
                return None, None
            return self._X[ind,], self._y[ind,]
        elif examples.ndim == 2:
            ind = []
            for i, exa in enumerate(examples):
                one_example_ind = self._find_one_example(exa)
                if one_example_ind == -1:
                    warnings.warn("Example %s for retrieving is not in the repository, skip." % str(exa),
                                  category=ValidityWarning)
                else:
                    ind.append(one_example_ind)
            return self._X[ind, ], self._y[ind, ]
        else:
            raise Exception("A 1D or 2D array is expected. But received: %d" % examples.ndim)

    def get_training_data(self):
        """Get training set.

        Returns
        -------
        X_train: list, shape (n_training_examples, n_features)
            The feature matrix of training data.

        y_train: list
            The labels of training data.

        indexes: list
            The indexes of the instances and labels.
            e.g. the first row of X_train is the indexes[0] instance in feature matrix X.
        """
        return copy.deepcopy(self._X), copy.deepcopy(self._y), np.asarray(self._indexes)

    def clear(self):
        """
            Clear the Matrix Knowledge Repository.
        """
        self.cost_inall = 0
        self._cost_arr = []
        self.num_of_queries = 0
        self._instance_flag = False
        self._X = None
        self._y = None
        self._indexes = None
        self._query_history.clear()

    def _update_query_history(self, labels, indexes, cost):
        """record the query history"""
        self._query_history.append(((labels, cost), indexes))

    def full_history(self):
        """return full version of query history"""
        tb = pt.PrettyTable()
        # tb.set_style(pt.MSWORD_FRIENDLY)
        for query_ind in range(len(self._query_history)):
            query_result = self._query_history[query_ind]
            tb.add_column(str(query_ind), ["query_index:%s\nresponse:%s\ncost:%s" % (
                str(query_result[1]), str(query_result[0][0]), str(query_result[0][1]))])
        tb.add_column('in all', ["number_of_queries:%s\ncost:%s" % (str(len(self._query_history)), str(self.cost_inall))])
        return str(tb)
