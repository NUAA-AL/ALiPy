"""
Heuristic:
1. Preset number of quiries
2. Preset limitation of cost
3. Preset percent of unlabel pool is labeled
4. Preset running time (CPU time) is reached
5. No unlabeled samples available
Formal (Not Implement yet):
5. The Performance of a learner has reached a plateau
6. The cost of acquiring new training data is greater than the cost of the errors made by the current model
"""

from __future__ import division
import time

import numpy as np

from .state_io import StateIO

__all__ = ['StoppingCriteria',
           ]

class StoppingCriteria:
    """Class to implement stopping criteria.

    Initialize it with a certain string to determine when to stop.

    It needs to collect the information in each iteration of active learning by the StateIO object.
    Once a fold of experiment is finished, you should reset the StoppingCriteria object to make
    it available to the other experiment fold.

    Example of supported stopping criteria:
    1. No unlabeled samples available (default)
    2. Preset number of queries is reached
    3. Preset limitation of cost is reached
    4. Preset percent of unlabel pool is labeled
    5. Preset running time (CPU time) is reached

    Parameters
    ----------
    stopping_criteria: str, optional (default=None)
        Stopping criteria, must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']

        None: Stop when no unlabeled samples available
        'num_of_queries': Stop when preset number of quiries is reached
        'cost_limit': Stop when cost reaches the limit.
        'percent_of_unlabel': Stop when specific percentage of unlabeled data pool is labeled.
        'time_limit': Stop when CPU time reaches the limit.

    value: {int, float}, optional (default=None)
        The value of the corresponding stopping criterion.
    """

    def __init__(self, stopping_criteria=None, value=None):
        if stopping_criteria not in [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']:
            raise ValueError("Stopping criteria must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']")
        self._stopping_criteria = stopping_criteria
        if isinstance(value, np.generic):
            # value = np.asscalar(value)    # deprecated in numpy v1.16
            value = value.item()

        if stopping_criteria == 'num_of_queries':
            if not isinstance(value, int):
                value = int(value)
        else:
            if not isinstance(value, float) and value is not None:
                value = float(value)
        if stopping_criteria == 'time_limit':
            self._start_time = time.clock()
        self.value = value

        # collect information
        self._current_iter = 0
        self._accum_cost = 0
        self._current_unlabel = 100
        self._percent = 0

        self._init_value = value

    def is_stop(self):
        """
            Determine whether termination conditions have been met,
            if so,return True.
        """
        if self._current_unlabel == 0:
            return True
        elif self._stopping_criteria == 'num_of_queries':
            if self._current_iter >= self.value:
                return True
            else:
                return False
        elif self._stopping_criteria == 'cost_limit':
            if self._accum_cost >= self.value:
                return True
            else:
                return False
        elif self._stopping_criteria == 'percent_of_unlabel':
            if self._percent >= self.value:
                return True
            else:
                return False
        elif self._stopping_criteria == 'time_limit':
            if time.clock() - self._start_time >= self.value:
                return True
            else:
                return False
        return False

    def update_information(self, saver):
        """update value according to the specific criterion.

        Parameters
        ----------
        saver: StateIO
            StateIO object
        """
        _,_,_, Uindex = saver.get_workspace()
        _, _, _, ini_Uindex = saver.get_workspace(iteration=0)
        self._current_unlabel = len(Uindex)
        if self._stopping_criteria == 'num_of_queries':
            self._current_iter = len(saver)
        elif self._stopping_criteria == 'cost_limit':
            self._accum_cost = saver.cost_inall
        elif self._stopping_criteria == 'percent_of_unlabel':
            self._percent = (len(ini_Uindex)-len(Uindex))/len(ini_Uindex)
        return self

    def reset(self):
        """
            Reset the current state to the initial.
        """
        self.value = self._init_value
        self._start_time = time.clock()
        self._current_iter = 0
        self._accum_cost = 0
        self._current_unlabel = 100
        self._percent = 0
