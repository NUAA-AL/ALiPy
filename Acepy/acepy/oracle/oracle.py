"""
Pre-defined oracle class
Implement classical situation
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import collections
import copy
import os
import random

import numpy as np
import prettytable as pt
from sklearn.utils.validation import check_array

from ..utils import interface
from ..index.multi_label_tools import check_index_multilabel
from ..utils.ace_warnings import *
from ..utils.misc import check_one_to_one_correspondence, unpack


class Oracle(interface.BaseVirtualOracle):
    """Oracle in active learning whose role is to label the given query.

    This class implements basic definition of oracle used in experiment.
    Oracle can provide information given both instance or index. The returned
    information is depending on specific scenario.

    Parameters
    ----------
    labels:  array-like
        label matrix. shape like [n_samples, n_classes] or [n_samples]

    examples: array-like, optional (default=None)
        array of _examples, initialize with this parameter to make
        "query by instance" available. Shape like [n_samples, n_features]

    indexes: array-like, optional (default=None)
        index of _examples, if not provided, it will be generated
        automatically started from 0.

    cost: array_like, optional (default=None)
        costs of each queried instance, should have the same length
        and is one-to-one correspondence of y, default is 1.
    """

    def __init__(self, labels, examples=None, indexes=None, cost=None):
        if not check_one_to_one_correspondence(labels, examples, indexes, cost):
            raise ValueError("Different length of parameters found. "
                             "All parameters should be list type with the same length")

        labels = check_array(labels, ensure_2d=False, dtype=None)
        if isinstance(labels[0], np.generic):
            self._label_type = type(np.asscalar(labels[0]))
        else:
            self._label_type = type(labels[0])
        self._label_dim = labels.ndim

        # check parameters
        self._indexes = indexes if indexes is not None else [i for i in range(len(labels))]
        self._cost_flag = True if cost is not None else False
        self._instance_flag = True if examples is not None else False

        # several _indexes construct
        if self._instance_flag:
            examples = [tuple(vec) for vec in examples]
            self._exa2ind = dict(zip(examples, self._indexes))
        if self._cost_flag:
            self._ind2all = dict(zip(self._indexes, zip(labels, cost)))
        else:
            self._ind2all = dict(zip(self._indexes, labels))

    @property
    def index_keys(self):
        return self._ind2all.keys()

    @property
    def example_keys(self):
        if self._instance_flag:
            return np.asarray(self._exa2ind.keys())
        else:
            return None

    def _add_one_entry(self, label, index, example=None, cost=None):
        """Adding entry to the oracle.

        Add new entry to the oracle for future querying where index is the queried elements,
        label is the returned data. Index should not be in the oracle. Example and cost should
        accord with the initializing (If exists in initializing, then must be provided.)

        The data provided must have the same type with the initializing data. If different, a
        transform is attempted.

        Parameters
        ----------
        label:  array-like
            Label matrix.

        index: object
            Index of examples, should not in the oracle.

        example: array-like, optional (default=None)
            Array of examples, initialize with this parameter to turn
            "query by instance" on.

        cost: array_like, optional (default=None)
            Cost of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        if isinstance(label, np.generic):
            label = np.asscalar(label)
        if isinstance(label, list):
            label = np.array(label)
        if not isinstance(label, self._label_type):
            raise TypeError("Different types of _labels found when adding entries: %s is expected but received: %s" %
                            (str(self._label_type), str(type(label))))
        if self._instance_flag:
            if example is None:
                raise Exception("This oracle has the instance information,"
                                "must provide example parameter when adding entry")
            self._exa2ind[example] = index
        if self._cost_flag:
            if cost is None:
                raise Exception("This oracle has the cost information,"
                                "must provide cost parameter when adding entry")
            self._ind2all[index] = (label, cost)
        else:
            self._ind2all[index] = label

    def add_knowledge(self, labels, indexes, examples=None, cost=None):
        """Adding entries to the oracle.

        Add new entries to the oracle for future querying where indexes are the queried elements,
        labels are the returned data. Indexes should not be in the oracle. Examples and cost should
        accord with the initializing (If exists in initializing, then must be provided.)

        Parameters
        ----------
        labels: array-like or object
            Label matrix.

        indexes: array-like or object
            Index of examples, should not in the oracle.
            if update multiple entries to the oracle, a list or np.ndarray type is expected,
            otherwise, it will be cheated as only one entry.

        examples: array-like, optional (default=None)
            Array of examples.

        cost: array_like, optional (default=None)
            Cost of each queried instance, should have the same length
            and is one-to-one correspondence of y, default is 1.
        """
        labels, indexes, examples, cost = unpack(labels, indexes, examples, cost)
        if not isinstance(indexes, (list, np.ndarray)):
            self._add_one_entry(labels, indexes, examples, cost)
        else:
            if not check_one_to_one_correspondence(labels, indexes, examples, cost):
                raise ValueError("Different length of parameters found.")
            for i in range(len(labels)):
                self._add_one_entry(labels[i], indexes[i], example=examples[i] if examples is not None else None,
                                    cost=cost[i] if cost is not None else None)

    def query_by_index(self, indexes):
        """Query function.

        Parameters
        ----------
        indexes: list or int
            Index to query, if only one index to query (batch_size = 1),
            an int is expected.

        Returns
        -------
        labels: list
            supervised information of queried index.

        cost: list
            corresponding cost produced by query.
        """
        if not isinstance(indexes, (list, np.ndarray)):
            indexes = [indexes]
        sup_info = []
        cost = []
        for k in indexes:
            if k in self._ind2all.keys():
                if self._cost_flag:
                    sup_info.append(self._ind2all[k][0])
                    cost.append(self._ind2all[k][1])
                else:
                    sup_info.append(self._ind2all[k])
                    cost.append(1)
            else:
                self._do_missing(k)
        return sup_info, cost

    def query_by_example(self, queried_examples):
        """Query function, query information giving an instance.
        Note that, this function only available if initializes with
        data matrix.

        Parameters
        ----------
        queried_examples: array_like
            [n_samples, n_features]

        Returns
        -------
        sup_info: list
            supervised information of queried instance.

        costs: list
            corresponding costs produced by query.
        """
        if not self._instance_flag:
            raise Exception("This oracle do not have the instance information, query_by_instance is not supported")
        if not isinstance(queried_examples, (list, np.ndarray)):
            raise TypeError("An list or numpy.ndarray is expected, but received:%s" % str(type(queried_examples)))
        if len(np.shape(queried_examples)) == 1:
            queried_examples = [queried_examples]
        q_id = []
        for k in queried_examples:
            k = tuple(k)
            if k in self._exa2ind.keys():
                q_id.append(self._exa2ind[k])
            else:
                self._do_missing(k, 'instance pool')
        return self.query_by_index(q_id)

    def _do_missing(self, key, dict_name='index pool'):
        """

        Parameters
        ----------
        key

        Returns
        -------

        """
        raise KeyError("%s is not in the " + dict_name + " of this oracle" % str(key))

    def __repr__(self):
        return str(self._ind2all)

    def save_oracle(self, saving_path):
        """Save the oracle to file.

        Parameters
        ----------
        saving_path: str
            path to save the settings. If a dir is provided, it will generate a file called
            'al_settings.pkl' for saving.

        """
        if saving_path is None:
            return
        else:
            if not isinstance(saving_path, str):
                raise TypeError("A string is expected, but received: %s" % str(type(saving_path)))
        import pickle
        saving_path = os.path.abspath(saving_path)
        if os.path.isdir(saving_path):
            f = open(os.path.join(saving_path, 'oracle.pkl'), 'wb')
        else:
            f = open(os.path.abspath(saving_path), 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load_oracle(cls, path):
        """Loading ToolBox object from path.

        Parameters
        ----------
        path: str
            Path to a specific file, not a dir.

        Returns
        -------
        setting: ToolBox
            Object of ToolBox.
        """
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        import pickle
        f = open(os.path.abspath(path), 'rb')
        setting_from_file = pickle.load(f)
        f.close()
        return setting_from_file


class OracleQueryInstance(Oracle):
    """Oracle to label all _labels of an instance.
    """


class OracleQueryMultiLabel(Oracle):
    """Oracle to label part of _labels of instance in multi-label setting.

    When initializing, a 2D array of _labels and cost of EACH label should be given.

    Parameters
    ----------
    labels:  array-like
        label matrix. Shape like [n_samples, n_classes]

    examples: array-like, optional (default=None)
        array of _examples, initialize with this parameter to make
        "query by instance" available. Shape like [n_samples, n_features]

    indexes: array-like, optional (default=None)
        index of _examples, if not provided, it will be generated
        automatically started from 0.

    cost: array_like, optional (default=None)
        cost of each labels, shape [n_classes] to specify a cost for each label.
        Or [n_samples, n_classes] to specify a fine-grained label cost.

    """

    def __init__(self, labels, examples=None, indexes=None, cost=None):
        labels = check_array(labels, ensure_2d=True, dtype=None)
        self._label_shape = np.shape(labels)
        if cost is not None:
            sp_cost = np.shape(cost)
            if len(sp_cost) == 1 and sp_cost[0] == self._label_shape[1]:
                self._fine_grained_cost = False
                self._label_cost = copy.copy(cost)
                cost = np.asarray(cost)
                cost = np.tile(cost, (self._label_shape[0], 1))
            else:
                self._fine_grained_cost = True
        super(OracleQueryMultiLabel, self).__init__(labels, examples, indexes, cost)

    def _add_one_entry(self, label, index, example=None, cost=None):
        """Adding entry to the oracle.

        Add new entry to the oracle for future querying where index is the queried elements,
        label is the returned data. Index should not be in the oracle. Example and cost should
        accord with the initializing (If exists in initializing, then must be provided.)

        Parameters
        ----------
        label:  array-like
            Label matrix.

        index: int
            Index of examples, should not in the oracle.

        example: array-like, optional (default=None)
            Array of examples.

        cost: array_like, optional (default=None)
            Cost of each queried instance.
        """
        if index in self._ind2all.keys():
            warnings.warn("The entry for adding has already exist in the oracle. Skip.")
            return

        if len(label) != self._label_shape[1]:
            raise ValueError(
                "Different dimension of labels found when adding entries: %s is expected but received: %s" %
                (str(self._label_shape[1]), str(len(label))))
        if self._instance_flag:
            if example is None:
                raise Exception("This oracle has the instance information,"
                                "must provide example parameter when adding entry")
            example = tuple(example)
            self._exa2ind[example] = index
        if self._cost_flag:
            if cost is None:
                if self._fine_grained_cost == False:
                    cost = copy.copy(self._label_cost)
                else:
                    raise Exception("This oracle has a fine-grained cost matrix, "
                                    "the cost of a new entry must be provided.")
            if len(cost) != self._label_shape[1]:
                raise ValueError(
                    "Different dimension of cost found when adding entries: %s is expected but received: %s" %
                    (str(self._label_shape[1]), str(len(cost))))
            self._ind2all[index] = (np.array(label), cost)
        else:
            self._ind2all[index] = np.array(label)

    def query_by_index(self, indexes):
        """Query function in multi-label setting

        In multi-label setting, a query index is a tuple.
        A single index should only have 1 element (example_index, ) to query all _labels or
        2 elements (example_index, [label_indexes]) to query specific _labels.
        A list of index can be provided.

        Parameters
        ----------
        indexes: list or tuple or int
            index to query, if only one index to query (batch_size = 1),a tuple or an int is expected.
            e.g., in 10 class classification setting, queried_index = (1, [3,4])
            means query the 2nd instance's 4th,5th _labels.
            some legal single index _examples:
            queried_index = (1, [3,4])
            queried_index = (1, [3])
            queried_index = (1, 3)
            queried_index = (1, (3))
            queried_index = (1, (3,4))
            queried_index = (1, )   # query all _labels

            One or more indexes could be provided.

        Returns
        -------
        sup_info: list
            supervised information of queried index.

        costs: list
            corresponding costs produced by query.
        """
        # check validity of the given indexes
        indexes = check_index_multilabel(indexes)

        # prepare queried _labels
        sup_info = []
        costs = []
        for k in indexes:
            # k is a tuple with 2 elements
            k_len = len(k)
            if k_len != 1 and k_len != 2:
                raise ValueError(
                    "A single index should only have 1 element (example_index, ) to query all _labels or"
                    "2 elements (example_index, [label_indexes]) to query specific _labels. But found %d in %s" %
                    (len(k), str(k)))
            example_ind = k[0]
            if k_len == 1:
                label_ind = [i for i in range(self._label_shape[1])]
            else:
                if isinstance(k[1], collections.Iterable):
                    label_ind = [i for i in k[1] if 0 <= i < self._label_shape[1]]
                else:
                    assert (0 <= k[1] < self._label_shape[1])
                    label_ind = [k[1]]

            # fetch data
            if example_ind in self._ind2all.keys():
                if self._cost_flag:
                    sup_info.append(self._ind2all[example_ind][0][label_ind])
                    costs.append(self._ind2all[example_ind][1][label_ind])
                else:
                    sup_info.append(self._ind2all[example_ind][label_ind])
                    costs.append(np.ones(len(label_ind)))
            else:
                self._do_missing(k)
        return sup_info, costs

    def query_by_example(self, examples):
        """Query function, query information giving an instance.

        Note that, this function only available if initializes with
        data matrix.

        In multi-label setting, a query index is a tuple.
        A single index should only have 1 element (feature_vector, ) to query all _labels or
        2 elements (feature_vector, [label_indexes]) to query specific _labels.
        A list of index can be provided.

        Parameters
        ----------
        examples: array_like
            [n_samples, n_features]

        Returns
        -------
        labels: list
            supervised information of queried instance.

        cost: list
            Corresponding cost produced by query.
        """
        if not self._instance_flag:
            raise Exception("This oracle do not have the instance information, query_by_example is not supported.")
        # check validity of the given examples
        if not isinstance(examples, (list, np.ndarray)):
            examples = [examples]

        q_id = []
        for k in examples:
            # k is a tuple with 2 elements
            k_len = len(k)
            if k_len != 1 and k_len != 2:
                raise ValueError(
                    "A single index should only have 1 element (feature_vector, ) to query all _labels or"
                    "2 elements (feature_vector, [label_indexes]) to query specific _labels. But found %d in %s" %
                    (len(k), str(k)))
            example_fea = tuple(k[0])

            # fetch data
            if example_fea in self._exa2ind.keys():
                if k_len == 1:
                    q_id.append((example_fea,))
                else:
                    q_id.append((example_fea, k[1]))
            else:
                self._do_missing(example_fea, 'instance pool')
        return self.query_by_index(q_id)


class Oracles:
    """Class to support crowdsourcing setting.

    This class is a container that support multiple oracles work together.
    It will store the cost in all and cost for each oracle for analysing.
    """

    def __init__(self):
        self._oracle_dict = dict()
        self.cost_inall = 0
        self.query_history = []

    def add_oracle(self, oracle_name, oracle_object):
        """Adding an oracle. The oracle name should be unique to identify
        different oracles.

        Parameters
        ----------
        oracle_name: str
            id of the oracle.

        oracle_object: utils.base.BaseOracle
            oracle object.
        """
        assert (isinstance(oracle_object, interface.BaseVirtualOracle))
        self._oracle_dict[oracle_name] = oracle_object
        return self

    def query_from(self, index_for_querying, oracle_name=None):
        """query index_for_querying from oracle_name.
        If oracle_name is not specified, it will query one of the oracles randomly.

        Parameters
        ----------
        index_for_querying: object
            index for querying.

        oracle_name: str, optional (default=None)
            query from which oracle. If not specified, it will query one of the
            oracles randomly.

        Returns
        -------
        sup_info: list
            supervised information of queried index.

        costs: list
            corresponding costs produced by query.
        """
        if oracle_name is None:
            oracle_name = random.sample(self._oracle_dict.keys(), 1)[0]
        result = self._oracle_dict[oracle_name].query_by_index(index_for_querying)

        self._update_query_history(oracle_name, result, index_for_querying)
        self.cost_inall += np.sum(result[1])
        return result

    def get_oracle(self, oracle_name):
        return self._oracle_dict[oracle_name]

    def _update_query_history(self, oracle_name, query_result, index_for_querying):
        """record the query history"""
        self.query_history.append((oracle_name, query_result, index_for_querying))

    def __repr__(self):
        """return summaries of each oracle.

        This function returns the content of this object.
        """
        # collect information for displaying
        # key: name
        # value: (query_times, cost_incured)
        display_dict = dict()
        for key in self._oracle_dict.keys():
            display_dict[key] = [0, 0]
        for query in self.query_history:
            # query is a triplet: (oracle_name, result, index_for_querying)
            # types of elements are: (str, [[_labels], [cost]], [indexes])
            display_dict[query[0]][0] += 1
            display_dict[query[0]][1] += np.sum([np.sum(query[1][1][i]) for i in range(len(query[1][1]))])

        tb = pt.PrettyTable()
        tb.field_names = ['oracles', 'number_of_labeling', 'cost']
        for key in display_dict.keys():
            tb.add_row([key, display_dict[key][0], display_dict[key][1]])
        return str(tb)

    def full_history(self):
        """return full version of query history"""
        oracle_name_list = list(self._oracle_dict.keys())
        oracles_num = len(oracle_name_list)
        oracle_labeling_count = [0] * oracles_num

        tb = pt.PrettyTable()
        # tb.set_style(pt.MSWORD_FRIENDLY)

        tb.add_column('oracles', oracle_name_list)
        for query_ind in range(len(self.query_history)):
            query_result = self.query_history[query_ind]
            name_ind = oracle_name_list.index(query_result[0])
            oracle_labeling_count[name_ind] += 1
            tb.add_column(str(query_ind), ['\\' if i != name_ind else "query_index:%s\nresponse:%s\ncost:%s" % (
            str(query_result[2]), str(query_result[1][0]), str(query_result[1][1])) for i in range(oracles_num)])

        tb.add_column('in all', oracle_labeling_count)
        return str(tb)


class OracleQueryFeatures(OracleQueryMultiLabel):
    """Oracle to give part of features of instance in feature querying setting.

    When initializing, a 2D array of feature matrix and cost of EACH feature should be given.

    Parameters
    ----------
    feature_mat: array-like
        array of _examples, initialize with this parameter to make
        "query by instance" available. Shape like [n_samples, n_features]

    indexes: array-like, optional (default=None)
        index of _examples, if not provided, it will be generated
        automatically started from 0.

    cost: array_like, optional (default=None)
        cost of each queried instance, should be one-to-one correspondence of each feature,
        default is all 1. Shape like [n_samples, n_classes]
    """
    def __init__(self, feature_mat, indexes=None, cost=None):
        super(OracleQueryFeatures, self).__init__(labels=feature_mat, indexes=indexes, cost=cost)
