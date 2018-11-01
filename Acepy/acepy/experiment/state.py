"""
State
Container to store all information in one AL iteration.
The information includes:
1. The performance after each query
2. The selected index for each query
3. Additional user-defined entry
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

import copy

import numpy as np

from acepy.utils.ace_warnings import *


class State:
    """A class to store information in one iteration of active learning
    for auditting and analysing.

    Parameters
    ----------
    select_index: array-like or object
        If multiple select_index are provided, it should be a list or np.ndarray type.
        otherwise, it will be treated as only one pair for adding.

    performance: array-like or object
        Performance after querying.

    queried_label: array-like or object, optional
        The queried label.

    cost: array-like or object, optional
        Cost corresponds to the query.
    """

    def __init__(self, select_index, performance, queried_label=None, cost=None):
        if not isinstance(select_index, (list, np.ndarray)):
            select_index = [select_index]

        self._save_seq = dict()
        self._save_seq['select_index'] = copy.deepcopy(select_index)
        self._save_seq['performance'] = copy.copy(performance)
        if queried_label is not None:
            self._save_seq['queried_label'] = copy.deepcopy(queried_label)
        if cost is not None:
            self._save_seq['cost'] = copy.copy(cost)
        self.batch_size = len(select_index)

    def __getitem__(self, item):
        return self.get_value(key=item)

    def __setitem__(self, key, value):
        return self.add_element(key=key, value=value)

    def keys(self):
        """Return the stored keys."""
        return self._save_seq.keys()

    def add_element(self, key, value):
        """Add an element to the object.

        Parameters
        ----------
        key: object
            Key to be added, should not in the object.

        value: object
            The value corresponds to the key.
        """
        self._save_seq[key] = copy.deepcopy(value)

    def del_element(self, key):
        """Deleting an element in the object.

        Parameters
        ----------
        key: object
            Key for deleting. Should not be one of the critical information:
            ['select_index', 'queried_info', 'performance', 'cost']
        """
        if key in ['select_index', 'queried_info', 'performance', 'cost']:
            warnings.warn("Critical information %s can not be discarded." % str(key),
                          category=ValidityWarning)
        elif key not in self._save_seq.keys():
            warnings.warn("Key %s to be discarded is not in the object, skip." % str(key),
                          category=ValidityWarning)
        else:
            self._save_seq.pop(key)

    def get_value(self, key):
        """Get a specific value given key."""
        return self._save_seq[key]

    def set_value(self, key, value):
        """Modify the value of an existed item.

        Parameters
        ----------
        key: object
            Key in the State, must a existed key

        value: object,
            Value to cover the original value
        """
        if key not in self._save_seq.keys():
            raise KeyError('key must be an existed one in State')
        self._save_seq[key] = copy.deepcopy(value)

    def __repr__(self):
        return self._save_seq.__repr__()
