"""
ABC for alipy
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.io as scio
import scipy.stats
from sklearn.utils.validation import check_X_y

from .ace_warnings import *


class BaseQueryStrategy:
    """Base query class.

    The parameters and global const are set in __init__()
    The instance to query can be obtained by select(), the labeled and unlabeled
    indexes of instances should be given. An array of selected elements in unlabeled indexes
    should be returned.

    Note that, the X, y in initializing is the whole data set.
    If a method needs to construct kernel matrix or something like that
    which uses the information of test set, the train_idx of the
    data set should be given in initializing.
    """

    __metaclass__ = ABCMeta

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
    def select(self, *args, **kwargs):
        """Select instances to query."""
        pass

        # def select_by_prediction_mat(self, unlabel_index, predict, **kwargs):
        #     """select in a model-independent way.
        #
        #     Parameters
        #     ----------
        #     prediction_mat: array, shape [n_examples, n_classes]
        #         The probability prediction matrix.
        #
        #     unlabel_index: {list, np.ndarray, IndexCollection}
        #         The indexes of unlabeled instances. Should be one-to-one
        #         correspondence to the prediction_mat
        #
        #     Returns
        #     -------
        #     selected_index: list
        #         The elements of selected_index should be in unlabel_index.
        #     """
        #     pass


class BaseVirtualOracle:
    """
    Basic class of virtual Oracle for experiment

    This class will build a dictionary between index-label in the __init__().
    When querying, the queried_index should be one of the key in the dictionary.
    And the label which corresponds to the key will be returned.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def query_by_index(self, indexes):
        """Return cost and queried info.

        Parameters
        ----------
        indexes: array
            Queried indexes.

        Returns
        -------
        Labels of queried indexes AND cost
        """
        pass


class BaseCollection:
    """The basic container of indexes.

    Functions include:
    1. Update new indexes.
    2. Discard existed indexes.
    3. Validity checking. (Repeated element, etc.)
    """

    __metaclass__ = ABCMeta
    _innercontainer = None
    _element_type = None

    def __contains__(self, other):
        return other in self._innercontainer

    def __iter__(self):
        return iter(self._innercontainer)

    def __len__(self):
        return len(self._innercontainer)

    def __repr__(self):
        return self._innercontainer.__repr__()

    @abstractmethod
    def add(self, *args):
        """Add element to the container."""
        pass

    @abstractmethod
    def discard(self, *args):
        """Discard element in the container."""
        pass

    @abstractmethod
    def update(self, *args):
        """Update multiple elements to the container."""
        pass

    @abstractmethod
    def difference_update(self, *args):
        """Discard multiple elements in the container"""
        pass

    def remove(self, value):
        """Remove an element. If not a member, raise a KeyError."""
        self.discard(value)

    def clear(self):
        """Clear the container."""
        self._innercontainer.clear()


class BaseRepository:
    """Base knowledge repository
    Store the information given by the oracle (labels, cost, etc.).

    Functions include:
    1. Retrieving
    2. History recording
    3. Get labeled set for training model
    """

    __metaclass__ = ABCMeta

    def __getitem__(self, index):
        """Same function with retrieve by index.

        Raise if item is not in the index set.

        Parameters
        ----------
        index: object
            Index of example and label.

        Returns
        -------
        example: np.ndarray
            The example corresponding the index.

        label: object
            The corresponding label of the index.
            The type of returned object is the same with the
            initializing.
        """
        return self.retrieve_by_indexes(index)

    @abstractmethod
    def add(self, select_index, label, cost=None, example=None):
        """Add an element to the repository."""
        pass

    @abstractmethod
    def discard(self, index=None, example=None):
        """Discard element either by index or example."""
        pass

    @abstractmethod
    def update_query(self, labels, indexes, cost=None, examples=None):
        """Updating data base with queried information."""
        pass

    @abstractmethod
    def retrieve_by_indexes(self, indexes):
        """Retrieve by indexes."""
        pass

    @abstractmethod
    def retrieve_by_examples(self, examples):
        """Retrieve by examples."""
        pass

    @abstractmethod
    def get_training_data(self):
        """Get training set."""
        pass

    @abstractmethod
    def clear(self):
        """Clear this container."""
        pass


class BaseAnalyser:
    """Base Analyser class for analysing experiment result.

    Functions include various validity checking and visualizing of the given data.

    Normally, the results should be a list which contains k elements. Each element represents
    one fold experiment result.
    Legal result object includes:
        - StateIO object.
        - A list contains n performances for n queries.
        - A list contains n tuples with 2 elements, in which, the first
          element is the x_axis (e.g., iteration, cost),
          and the second element is the y_axis (e.g., the performance)
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        # The data extracted from the original data.
        self._data_extracted = dict()
        # The summary of the data (each entry is optional according to the type of data):
        # 1. length
        # 2. batch_size
        # 3. performance mean and std
        # 4. cost
        self._data_summary = dict()

    def get_methods_names(self):
        return self.__data_raw.keys()

    def get_extracted_data(self, method_name):
        return self._data_extracted[method_name]

    @abstractmethod
    def add_method(self, method_results, method_name):
        """Add the results of a method."""
        pass

    @abstractmethod
    def plot_learning_curves(self, *args, **kwargs):
        """Plot the performance curves of different methods."""
        pass

    # some commonly used tool function for experiment analysing.
    @classmethod
    def paired_ttest(cls, a, b, alpha=0.05):
        """Performs a two-tailed paired t-test of the hypothesis that two
        matched samples, in the arrays a and b, come from distributions with
        equal means. The difference a-b is assumed to come from a normal
        distribution with unknown variance.  a and b must have the same length.

        Parameters
        ----------
        a: array-like
            array for paired t-test.

        b: array-like
            array for paired t-test.

        alpha: float, optional (default=0.05)
            A value alpha between 0 and 1 specifying the
            significance level as (100*alpha)%. Default is
            0.05 for 5% significance.

        Returns
        -------
        H: int
            the result of the test.
            H=0     -- indicates that the null hypothesis ("mean is zero")
                    cannot be rejected at the alpha% significance level
                    (No significant difference between a and b).
            H=1     -- indicates that the null hypothesis can be rejected at the alpha% level
                    (a and b have significant difference).

        Examples
        -------
        >>> from alipy.experiment.experiment_analyser import ExperimentAnalyser
        >>> a = [1.2, 2, 3]
        >>> b = [1.6, 2.5, 1.1]
        >>> print(ExperimentAnalyser.paired_ttest(a, b))
        1
        """
        rava = a
        ravb = b
        # check a,b
        sh = np.shape(a)
        if len(sh) == 1:
            pass
        elif sh[0] == 1 or sh[1] == 1:
            rava = np.ravel(a)
            ravb = np.ravel(b)
        else:
            raise Exception("a or b must be a 1-D array. but received: %s" % str(sh))
        assert (len(a) == len(b))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, pvalue = scipy.stats.ttest_rel(rava, ravb)
        H = int(pvalue <= alpha)
        return H

    @classmethod
    def load_matlab_file(cls, file_name):
        """load a data file in .mat format

        Parameters
        ----------
        file_name: str
            path to a matlab file

        Returns
        -------
        data: dict
            dictionary with variable names as keys, and loaded matrices as
            values.
        """
        return scio.loadmat(file_name)
