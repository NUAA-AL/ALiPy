"""
Pre-defined query strategy for noisy oracles.
There are 2 categories of methods.
1. Query from a single selected oracle.
    1.1 Always query from the best oracle
    1.2 Query from the most appropriate oracle
        according to the selected instance and label.
2. Query from multiple noisy oracles. Labels are obtained from multiple noisy oracles.
And the algorithm tries to obtain the accurate label for each instance.

Implement method:
1: CEAL (IJCAI'17)
2: IEthresh (KDD'09 Donmez)
Baselines:
Majority vote
Query from all oracles and majority vote
Random select an oracle
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import collections
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .base import BaseNoisyOracleQuery
from .query_labels import QueryInstanceUncertainty
from .query_labels import _get_proba_pred
from ..oracle import Oracles, Oracle


def majority_vote(labels, weight=None):
    """Perform majority vote to determine the true label from
    multiple noisy oracles.

    Parameters
    ----------
    labels: list
        A list with length=k, which contains the labels provided by
        k noisy oracles.

    weight: list, optional (default=None)
        The weights of each oracle. It should have the same length with
        labels.

    Returns
    -------
    vote_count: int
        The number of votes.

    vote_result: object
        The label of the selected_instance, produced by majority voting
        of the selected oracles.
    """
    oracle_weight = np.ones(len(labels)) if weight is None else weight
    assert len(labels) == len(oracle_weight)

    vote_result = collections.Counter(labels)
    most_votes = vote_result.most_common(n=1)
    return most_votes[0][0], most_votes[0][1]


def get_majority_vote(selected_instance, oracles):
    """Get the majority vote results of the selected instance.

    Parameters
    ----------
    selected_instance: int
        The indexes of selected samples. Should be a member of unlabeled set.

    oracles: {list, acepy.oracle.Oracles}
        An acepy.oracle.Oracle object that contains all the
        available oracles or a list of oracles.
        Each oracle should be a acepy.oracle.Oracle object.

    Returns
    -------
    selected_instance: int
        The index of selected instance. Selected by uncertainty.
    """
    if isinstance(oracles, list):
        oracle_type = 'list'
        for oracle in oracles:
            assert isinstance(oracle, Oracle)
    elif isinstance(oracles, Oracles):
        oracle_type = 'oracles'
    else:
        raise TypeError("The type of parameter oracles must be a list or acepy.oracle.Oracles object.")
    labeling_results = []
    for i in oracles.names() if oracle_type == 'oracles' else range(len(oracles)):
        lab, _ = oracles[i].query_by_index(selected_instance)
        labeling_results.append(lab[0])
    majority_vote_result = majority_vote(labeling_results)
    return majority_vote_result


class QueryNoisyOraclesCEAL(BaseNoisyOracleQuery):
    """Cost-Effective Active Learning from Diverse Labelers (CEAL) method assume
    that different oracles have different expertise. Even the very noisy oracle
    may perform well on some kind of examples. The cost of a labeler is proportional
    to its overall labeling quality and it is thus necessary to query from the right oracle
    according to the selected instance.

    This method will select an instance-labeler pair (x, a), and queries the label of x
    from a, where the selection of both the instance and labeler is based on a
    evaluation function Q(x, a).

    The selection of instance is depend on its uncertainty. The selection of oracle is
    depend on the oracle's performance on the nearest neighbors of selected instance.
    The cost of each oracle is proportional to its overall labeling quality.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    oracles: {list, acepy.oracle.Oracles}
        An acepy.oracle.Oracle object that contains all the
        available oracles or a list of oracles.
        Each oracle should be a acepy.oracle.Oracle object.

    initial_labeled_indexes: {list, np.ndarray, IndexCollection}
            The indexes of initially labeled samples. Used for initializing the scores of each oracle.

    References
    ----------
    [1] Sheng-Jun Huang, Jia-Lve Chen, Xin Mu, Zhi-Hua Zhou. 2017.
        Cost-Effective Active Learning from Diverse Labelers. In The
        Proceedings of the 26th International Joint Conference
        on Artificial Intelligence (IJCAI-17), 1879-1885.
    """

    def __init__(self, X, y, oracles, initial_labeled_indexes):
        super(QueryNoisyOraclesCEAL, self).__init__(X, y, oracles=oracles)
        # ytype = type_of_target(self.y)
        # if 'multilabel' in ytype:
        #     warnings.warn("This query strategy does not support multi-label.",
        #                   category=FunctionWarning)
        assert (isinstance(initial_labeled_indexes, collections.Iterable))
        self._ini_ind = np.asarray(initial_labeled_indexes)
        # construct a nearest neighbor object implemented by scikit-learn
        self._nntree = NearestNeighbors(metric='euclidean')
        self._nntree.fit(self.X[self._ini_ind])

    def select(self, label_index, unlabel_index, model=None, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.

        Returns
        -------
        selected_instance: int
            The index of selected instance.

        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """

        if model is None:
            model = LogisticRegression()
            model.fit(self.X[label_index], self.y[label_index])
        pred_unlab, _ = _get_proba_pred(self.X[unlabel_index], model)

        return self.select_by_prediction_mat(label_index, unlabel_index, pred_unlab,
                                             n_neighbors=kwargs.pop('n_neighbors', 10))

    def select_by_prediction_mat(self, label_index, unlabel_index, predict, **kwargs):
        """Query from oracles. Return the index of selected instance and oracle.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.

        Returns
        -------
        selected_instance: int
            The index of selected instance.

        selected_oracle: int or str
            The index of selected oracle.
            If a list is given, the index of oracle will be returned.
            If a Oracles object is given, the oracle name will be returned.
        """
        Q_table, oracle_ind_name_dict = self._calc_Q_table(label_index, unlabel_index, self._oracles, predict,
                                                           n_neighbors=kwargs.pop('n_neighbors', 10))
        # get the instance-oracle pair
        selected_pair = np.unravel_index(np.argmax(Q_table, axis=None), Q_table.shape)
        return [unlabel_index[selected_pair[1]]], oracle_ind_name_dict[selected_pair[0]]

    def _calc_Q_table(self, label_index, unlabel_index, oracles, pred_unlab, n_neighbors=10):
        """Query from oracles. Return the Q table and the oracle name/index of each row of Q_table.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        oracles: {list, acepy.oracle.Oracles}
            An acepy.oracle.Oracle object that contains all the
            available oracles or a list of oracles.
            Each oracle should be a acepy.oracle.Oracle object.

        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        n_neighbors: int, optional (default=10)
            How many neighbors of the selected instance will be used
            to evaluate the oracles.

        Returns
        -------
        Q_table: 2D array
            The Q table.

        oracle_ind_name_dict: dict
            The oracle name/index of each row of Q_table.
        """
        # Check parameter and initialize variables
        if self.X is None or self.y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        num_of_neighbors = n_neighbors
        if len(unlabel_index) <= 1:
            return unlabel_index

        Q_table = np.zeros((len(oracles), len(unlabel_index)))  # row:oracle, col:ins
        spv = np.shape(pred_unlab)
        # calc least_confident
        rx = np.partition(pred_unlab, spv[1] - 1, axis=1)
        rx = 1 - rx[:, spv[1] - 1]

        for unlab_ind, unlab_ins_ind in enumerate(unlabel_index):
            # evaluate oracles for each instance
            nn_dist, nn_of_selected_ins = self._nntree.kneighbors(X=self.X[unlab_ins_ind].reshape(1, -1),
                                                                  n_neighbors=num_of_neighbors,
                                                                  return_distance=True)
            nn_dist = nn_dist[0]
            nn_of_selected_ins = nn_of_selected_ins[0]
            nn_of_selected_ins = self._ini_ind[nn_of_selected_ins]  # map to the original population
            oracles_score = []
            oracles_cost = []
            for ora_ind, ora_name in enumerate(self._oracles_iterset):
                # calc q_i(x), expertise of this instance
                oracle = oracles[ora_name]
                labels, cost = oracle.query_by_index(nn_of_selected_ins)
                oracles_score.append(sum([nn_dist[i] * (labels[i] == self.y[nn_of_selected_ins[i]]) for i in
                                          range(num_of_neighbors)]) / num_of_neighbors)
                # calc c_i, cost of each labeler
                labels, cost = oracle.query_by_index(label_index)
                oracles_cost.append(
                    sum([labels[i] == self.y[label_index[i]] for i in range(len(label_index))]) / len(label_index))
                Q_table[ora_ind, unlab_ind] = oracles_score[ora_ind] * rx[unlab_ind] / oracles_cost[ora_ind]

        return Q_table, self._oracle_ind_name_dict


class QueryNoisyOraclesSelectInstanceUncertainty(BaseNoisyOracleQuery, metaclass=ABCMeta):
    """This class implement select and select_by_prediction_mat by uncertainty."""

    def __init__(self, X=None, y=None, oracles=None):
        super(QueryNoisyOraclesSelectInstanceUncertainty, self).__init__(X=X, y=y, oracles=oracles)

    def select(self, label_index, unlabel_index, model=None, **kwargs):
        """Select an instance and a batch of oracles to label it.
        The instance is selected by uncertainty, the oracles is
        selected by the difference between their
        labeling results and the majority vote results.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        Returns
        -------
        selected_instance: int
            The index of selected instance. Selected by uncertainty.

        selected_oracles: list
            The selected oracles for querying.
        """
        if model is None:
            model = LogisticRegression()
            model.fit(self.X[label_index], self.y[label_index])
        pred_unlab, _ = _get_proba_pred(self.X[unlabel_index], model)

        return self.select_by_prediction_mat(label_index, unlabel_index, pred_unlab)

    def select_by_prediction_mat(self, label_index, unlabel_index, predict):
        """Query from oracles. Return the index of selected instance and oracle.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        predict: : 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        Returns
        -------
        selected_instance: int
            The index of selected instance. Selected by uncertainty.

        selected_oracles: list
            The selected oracles for querying.
        """
        # Check parameter and initialize variables
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= 1:
            return unlabel_index

        # select instance and oracle
        unc = QueryInstanceUncertainty(measure='least_confident')
        selected_instance = unc.select_by_prediction_mat(unlabel_index=unlabel_index, predict=predict, batch_size=1)[0]
        return [selected_instance], self.select_by_given_instance(selected_instance)

    @abstractmethod
    def select_by_given_instance(self, selected_instance):
        pass


class QueryNoisyOraclesIEthresh(QueryNoisyOraclesSelectInstanceUncertainty):
    """IEthresh will select a batch of oracles to label the selected instance.
    It will score for each oracle according to the difference between their
    labeling results and the majority vote results.

    At each iteration, a batch of oracles whose scores are larger than a threshold will be selected.
    Oracle with a higher score is more likely to be selected.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    oracles: {list, acepy.oracle.Oracles}
        An acepy.oracle.Oracle object that contains all the
        available oracles or a list of oracles.
        Each oracle should be a acepy.oracle.Oracle object.

    initial_labeled_indexes: {list, np.ndarray, IndexCollection}
            The indexes of initially labeled samples. Used for initializing the scores of each oracle.

    epsilon: float, optional (default=0.1)
            The value to determine how many oracles will be selected.
            S_t = {a|UI(a) >= epsilon * max UI(a)}

    References
    ----------
    [1] Donmez P , Carbonell J G , Schneider J . Efficiently learning the accuracy of labeling
    sources for selective sampling.[C] ACM SIGKDD International Conference on
    Knowledge Discovery & Data Mining. ACM, 2009.
    """

    def __init__(self, X, y, oracles, initial_labeled_indexes, **kwargs):
        super(QueryNoisyOraclesIEthresh, self).__init__(X, y, oracles=oracles)
        self._ini_ind = np.asarray(initial_labeled_indexes)
        # record the labeling history of each oracle
        self._oracles_history = [dict()] * len(self._oracles_iterset)
        # record the results of majority vote
        self._majority_vote_results = dict()
        # calc initial QI(a) for each oracle a
        self._UI = np.ones(len(self._oracles_iterset))
        self.epsilon = kwargs.pop('epsilon', 0.1)

    def _calc_uia(self, oracle_history, majority_vote_result, alpha=0.05):
        """Calculate the UI(a) by providing the labeling history and the majority vote results.

        Parameters
        ----------
        oracle_history: dict
            The labeling history of an oracle. The key is the index of instance, the value is the
            label given by the oracle.

        majority_vote_result: dict
            The results of majority vote of instances. The key is the index of instance,
            the value is the label given by the oracle.

        alpha: float, optional (default=0.05)
            Used for calculating the critical value for the Student’s t-distribution with n−1
            degrees of freedom at the alpha/2 confidence level.

        Returns
        -------
        uia: float
            The UI(a) value.
        """
        n = len(self._oracles_iterset)
        t_crit_val = scipy.stats.t.isf([alpha / 2], n - 1)[0]
        reward_arr = []
        for ind in oracle_history.keys():
            if oracle_history[ind] == majority_vote_result[ind][1]:
                reward_arr.append(1)
            else:
                reward_arr.append(0)
        mean_a = np.mean(reward_arr)
        std_a = np.std(reward_arr)
        uia = mean_a + t_crit_val * std_a / np.sqrt(n)
        return uia

    def select_by_given_instance(self, selected_instance):
        """Select oracle to query by providing the index of selected instance.

        Parameters
        ----------
        selected_instance: int
            The indexes of selected samples. Should be a member of unlabeled set.

        Returns
        -------
        selected_oracles: list
            The selected oracles for querying.
        """
        selected_oracles = np.nonzero(self._UI > self.epsilon * np.max(self._UI))
        selected_oracles = selected_oracles[0]

        # update UI(a) for each selected oracle
        labeling_results = []
        for i in selected_oracles:
            lab, _ = self._oracles[self._oracle_ind_name_dict[i]].query_by_index(selected_instance)
            labeling_results.append(lab[0])
            self._oracles_history[i][selected_instance] = lab[0]
        majority_vote_result = majority_vote(labeling_results)
        reward_arr = np.zeros(len(selected_oracles))
        same_ind = np.nonzero(labeling_results == majority_vote_result)
        reward_arr[same_ind] = 1
        self._majority_vote_results[selected_instance] = majority_vote_result
        for i in selected_oracles:
            self._UI[i] = self._calc_uia(self._oracles_history[i], self._majority_vote_results)

        # return results
        return [self._oracle_ind_name_dict[i] for i in selected_oracles]


class QueryNoisyOraclesAll(QueryNoisyOraclesSelectInstanceUncertainty):
    """This strategy will select instance by uncertainty and query from all
    oracles and return the majority vote result.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    oracles: {list, acepy.oracle.Oracles}
        An acepy.oracle.Oracle object that contains all the
        available oracles or a list of oracles.
        Each oracle should be a acepy.oracle.Oracle object.
    """

    def __init__(self, oracles, X=None, y=None):
        super(QueryNoisyOraclesAll, self).__init__(X=X, y=y, oracles=oracles)

    def select_by_given_instance(self, selected_instance):
        """Select oracle to query by providing the index of selected instance.

        Parameters
        ----------
        selected_instance: int
            The indexes of selected samples. Should be a member of unlabeled set.

        Returns
        -------
        oracles_ind: list
            The indexes of selected oracles.
        """
        return self._oracle_ind_name_dict.values()


class QueryNoisyOraclesRandom(QueryNoisyOraclesSelectInstanceUncertainty):
    """Select an instance based on uncertainty and a random oracle to query."""

    def __init__(self, oracles, X=None, y=None):
        super(QueryNoisyOraclesRandom, self).__init__(X=X, y=y, oracles=oracles)

    def select_by_given_instance(self, selected_instance):
        return self._oracle_ind_name_dict[np.random.randint(0, len(self._oracles), 1)[0]]
