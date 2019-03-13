# -*- coding: UTF-8 -*-

"""
Pre-defined classical query strategy.

References:
[1] Settles, B. 2009. Active learning literature survey. Technical
    report, University of Wisconsin-Madison.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
from __future__ import print_function

import collections
import copy
import os

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.multiclass import unique_labels

from .base import BaseIndexQuery
from ..utils.ace_warnings import *
from ..utils.misc import nsmallestarg, randperm, nlargestarg


def _get_proba_pred(unlabel_x, model):
    """Get the probabilistic prediction results of the unlabeled set.

    Parameters
    ----------
    unlabel_x: 2d array
        Feature matrix of the unlabeled set.

    model: object
        Model object which has the method predict_proba.

    Returns
    -------
    pv: np.ndarray
        Probability predictions matrix with shape [n_samples, n_classes].

    spv: tuple
        Shape of pv.
    """
    if not hasattr(model, 'predict_proba'):
        raise Exception('model object must implement predict_proba methods in current algorithm.')
    pv = model.predict_proba(unlabel_x)
    pv = np.asarray(pv)
    spv = np.shape(pv)
    if len(spv) != 2 or spv[1] == 1:
        raise Exception('2d array with [n_samples, n_class] is expected, but received: \n%s' % str(pv))
    return pv, spv


class QueryInstanceUncertainty(BaseIndexQuery):
    """Uncertainty query strategy.
    The implement of uncertainty measure includes:
    1. margin sampling
    2. least confident
    3. entropy

    The above measures need the probabilistic output of the model.
    There are 3 ways to select instances in the data set.
    1. use select if you are using sklearn model.
    2. use the default logistic regression model to choose the instances
       by passing None to the model parameter.
    3. use select_by_prediction_mat by providing the probabilistic prediction
       matrix of your own model, shape [n_samples, n_classes].

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    measure: str, optional (default='entropy')
        measurement to calculate uncertainty, should be one of
        ['least_confident', 'margin', 'entropy', 'distance_to_boundary']
        --'least_confident' x* = argmax 1-P(y_hat|x) ,where y_hat = argmax P(yi|x)
        --'margin' x* = argmax P(y_hat1|x) - P(y_hat2|x), where y_hat1 and y_hat2 are the first and second
            most probable class labels under the model, respectively.
        --'entropy' x* = argmax -sum(P(yi|x)logP(yi|x))
        --'distance_to_boundary' Only available in binary classification, x* = argmin |f(x)|,
            your model should have 'decision_function' method which will return a 1d array.

    """

    def __init__(self, X=None, y=None, measure='entropy'):
        if measure not in ['least_confident', 'margin', 'entropy', 'distance_to_boundary']:
            raise ValueError("measure must be one of ['least_confident', 'margin', 'entropy', 'distance_to_boundary']")
        self.measure = measure
        super(QueryInstanceUncertainty, self).__init__(X, y)

    def select(self, label_index, unlabel_index, model=None, batch_size=1):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        # get unlabel_x
        if self.X is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        if model is None:
            model = LogisticRegression(solver='liblinear')
            model.fit(self.X[label_index if isinstance(label_index, (list, np.ndarray)) else label_index.index],
                      self.y[label_index if isinstance(label_index, (list, np.ndarray)) else label_index.index])
        unlabel_x = self.X[unlabel_index, :]
        ##################################
        if self.measure == 'distance_to_boundary':
            if not hasattr(model, 'decision_function'):
                raise TypeError(
                    'model object must implement decision_function methods in distance_to_boundary measure.')
            pv = np.absolute(model.decision_function(unlabel_x))
            spv = np.shape(pv)
            assert (len(spv) in [1, 2])
            if len(spv) == 2:
                if spv[1] != 1:
                    raise Exception('1d or 2d with 1 column array is expected, but received: \n%s' % str(pv))
                else:
                    pv = np.absolute(np.array(pv).flatten())
            return unlabel_index[nsmallestarg(pv, batch_size)]

        pv, _ = _get_proba_pred(unlabel_x, model)
        return self.select_by_prediction_mat(unlabel_index=unlabel_index, predict=pv,
                                             batch_size=batch_size)

    def select_by_prediction_mat(self, unlabel_index, predict, batch_size=1):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.

        predict: 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        pv = np.asarray(predict)  # predict value
        spv = np.shape(pv)  # shape of predict value

        if self.measure == 'distance_to_boundary':
            assert (len(spv) in [1, 2])
            if len(spv) == 2:
                if spv[1] != 1:
                    raise Exception('1d or 2d with 1 column array is expected, but received: \n%s' % str(pv))
                else:
                    pv = np.absolute(np.array(pv).flatten())
            else:
                pv = np.absolute(pv)
            return unlabel_index[nsmallestarg(pv, batch_size)]

        if len(spv) != 2 or spv[1] == 1:
            raise Exception('2d array with the shape [n_samples, n_classes]'
                            ' is expected, but received shape: \n%s' % str(spv))

        if self.measure == 'entropy':
            # calc entropy
            pv[pv <= 0] = 1e-06  # avoid zero division
            entro = [-np.sum(vec * np.log(vec)) for vec in pv]
            assert (len(np.shape(entro)) == 1)
            return unlabel_index[nlargestarg(entro, batch_size)]

        if self.measure == 'margin':
            # calc margin
            pat = np.partition(pv, (spv[1] - 2, spv[1] - 1), axis=1)
            pat = pat[:, spv[1] - 2] - pat[:, spv[1] - 1]
            return unlabel_index[nlargestarg(pat, batch_size)]

        if self.measure == 'least_confident':
            # calc least_confident
            pat = np.partition(pv, spv[1] - 1, axis=1)
            pat = 1 - pat[:, spv[1] - 1]
            return unlabel_index[nlargestarg(pat, batch_size)]

    @classmethod
    def calc_entropy(cls, predict_proba):
        """Calc the entropy for each instance.

        Parameters
        ----------
        predict_proba: array-like, shape [n_samples, n_class]
            Probability prediction for each instance.

        Returns
        -------
        entropy: list
            1d array, entropy for each instance.
        """
        pv = np.asarray(predict_proba)
        spv = np.shape(pv)
        if len(spv) != 2 or spv[1] == 1:
            raise Exception('2d array with the shape [n_samples, n_classes]'
                            ' is expected, but received shape: \n%s' % str(spv))
        # calc entropy
        entropy = [-np.sum(vec * np.log(vec)) for vec in pv]
        return entropy


class QueryRandom(BaseIndexQuery):
    """
    Randomly sample a batch of indexes from the unlabel indexes.
    The random strategy has been re-named to QueryInstanceRandom,
    this class will be deleted in v1.0.5.
    """
    def __init__(self, X=None, y=None):
        warnings.warn("QueryRandom will be deleted in the future. Use QueryInstanceRandom instead.",
                      category=DeprecationWarning)
        super(QueryRandom, self).__init__(X, y)

    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Select indexes randomly.

        Parameters
        ----------
        label_index: object
            Add this parameter to ensure the consistency of api of strategies.
            Please ignore it.

        unlabel_index: collections.Iterable
            The indexes of unlabeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        if len(unlabel_index) <= batch_size:
            return np.array([i for i in unlabel_index])
        perm = randperm(len(unlabel_index) - 1, batch_size)
        tpl = list(unlabel_index.index)
        return [tpl[i] for i in perm]


class QueryInstanceRandom(BaseIndexQuery):
    """Randomly sample a batch of indexes from the unlabel indexes."""

    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Select indexes randomly.

        Parameters
        ----------
        label_index: object
            Add this parameter to ensure the consistency of api of strategies.
            Please ignore it.

        unlabel_index: collections.Iterable
            The indexes of unlabeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        if len(unlabel_index) <= batch_size:
            return np.array([i for i in unlabel_index])
        perm = randperm(len(unlabel_index) - 1, batch_size)
        tpl = list(unlabel_index.index)
        return [tpl[i] for i in perm]


class QueryInstanceQBC(BaseIndexQuery):
    """The Query-By-Committee (QBC) algorithm.

    QBC minimizes the version space, which is the set of hypotheses that are consistent
    with the current labeled training data.

    This class implement the query-by-bagging method. Which uses the bagging in sklearn to
    construct the committee. So your model should be a sklearn model.
    If not, you may using the default logistic regression model by passing None model.

    There are 3 ways to select instances in the data set.
    1. use select if you are using sklearn model.
    2. use the default logistic regression model to choose the instances
       by passing None to the model parameter.
    3. use select_by_prediction_mat by providing the prediction matrix for each committee.
       Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
       or [n_samples] for class output.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    method: str, optional (default=query_by_bagging)
        Method name. This class only implement query_by_bagging for now.

    disagreement: str
        method to calculate disagreement of committees. should be one of ['vote_entropy', 'KL_divergence']

    References
    ----------
    [1] H.S. Seung, M. Opper, and H. Sompolinsky. Query by committee.
        In Proceedings of the ACM Workshop on Computational Learning Theory,
        pages 287–294, 1992.

    [2] N. Abe and H. Mamitsuka. Query learning strategies using boosting and bagging.
        In Proceedings of the International Conference on Machine Learning (ICML),
        pages 1–9. Morgan Kaufmann, 1998.
    """

    def __init__(self, X=None, y=None, method='query_by_bagging', disagreement='vote_entropy'):
        self._method = method
        super(QueryInstanceQBC, self).__init__(X, y)
        if disagreement in ['vote_entropy', 'KL_divergence']:
            self._disagreement = disagreement
        else:
            raise ValueError("disagreement must be one of ['vote_entropy', 'KL_divergence']")

    def select(self, label_index, unlabel_index, model=None, batch_size=1, n_jobs=None):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        batch_size: int, optional (default=1)
            Selection batch size.

        n_jobs: int, optional (default=None)
            How many threads will be used in training bagging.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        # get unlabel_x
        if self.X is None or self.y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        if model is None:
            model = LogisticRegression(solver='liblinear')
            model.fit(self.X[label_index],
                      self.y[label_index])

        unlabel_x = self.X[unlabel_index]
        label_x = self.X[label_index]
        label_y = self.y[label_index]
        #####################################

        # bagging
        if n_jobs is None:
            bagging = BaggingClassifier(model)
        else:
            bagging = BaggingClassifier(model, n_jobs=n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bagging.fit(label_x, label_y)
        est_arr = bagging.estimators_

        # calc score
        if self._disagreement == 'vote_entropy':
            score = self.calc_vote_entropy([estimator.predict(unlabel_x) for estimator in est_arr])
        else:
            score = self.calc_avg_KL_divergence([estimator.predict_proba(unlabel_x) for estimator in est_arr])
        return unlabel_index[nlargestarg(score, batch_size)]

    def select_by_prediction_mat(self, unlabel_index, predict, batch_size=1):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples. Should be one-to-one
            correspondence to the prediction matrix.

        predict: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        if self._disagreement == 'vote_entropy':
            score = self.calc_vote_entropy(predict)
        else:
            score = self.calc_avg_KL_divergence(predict)
        return unlabel_index[nlargestarg(score, batch_size)]

    def _check_committee_results(self, predict_matrices):
        """check the validity of given committee predictions.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        input_shape: tuple
            The shape of the predict_matrix

        committee_size: int
            The number of committees.

        """
        shapes = [np.shape(X) for X in predict_matrices if X is not None]
        uniques = np.unique(shapes, axis=0)
        if len(uniques) > 1:
            raise Exception("Found input variables with inconsistent numbers of"
                            " shapes: %r" % [int(l) for l in shapes])
        committee_size = len(predict_matrices)
        if not committee_size > 1:
            raise ValueError("Two or more committees are expected, but received: %d" % committee_size)
        input_shape = uniques[0]
        return input_shape, committee_size

    @classmethod
    def calc_vote_entropy(cls, predict_matrices):
        """Calculate the vote entropy for measuring the level of disagreement in QBC.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        score: list
            Score for each instance. Shape [n_samples]

        References
        ----------
        [1] I. Dagan and S. Engelson. Committee-based sampling for training probabilistic
            classifiers. In Proceedings of the International Conference on Machine
            Learning (ICML), pages 150–157. Morgan Kaufmann, 1995.
        """
        score = []
        input_shape, committee_size = cls()._check_committee_results(predict_matrices)
        if len(input_shape) == 2:
            ele_uni = np.unique(predict_matrices)
            if not (len(ele_uni) == 2 and 0 in ele_uni and 1 in ele_uni):
                raise ValueError("The predicted label matrix must only contain 0 and 1")
            # calc each instance
            for i in range(input_shape[0]):
                instance_mat = np.array([X[i, :] for X in predict_matrices if X is not None])
                voting = np.sum(instance_mat, axis=0)
                tmp = 0
                # calc each label
                for vote in voting:
                    if vote != 0:
                        tmp += vote / len(predict_matrices) * np.log(vote / len(predict_matrices))
                score.append(-tmp)
        else:
            input_mat = np.array([X for X in predict_matrices if X is not None])
            # label_arr = np.unique(input_mat)
            # calc each instance's score
            for i in range(input_shape[0]):
                count_dict = collections.Counter(input_mat[:, i])
                tmp = 0
                for key in count_dict:
                    tmp += count_dict[key] / committee_size * np.log(count_dict[key] / committee_size)
                score.append(-tmp)
        return score

    @classmethod
    def calc_avg_KL_divergence(cls, predict_matrices):
        """Calculate the average Kullback-Leibler (KL) divergence for measuring the
        level of disagreement in QBC.

        Parameters
        ----------
        predict_matrices: list
            The prediction matrix for each committee.
            Each committee predict matrix should have the shape [n_samples, n_classes] for probabilistic output
            or [n_samples] for class output.

        Returns
        -------
        score: list
            Score for each instance. Shape [n_samples]

        References
        ----------
        [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning for
            text classification. In Proceedings of the International Conference on Machine
            Learning (ICML), pages 359–367. Morgan Kaufmann, 1998.
        """
        score = []
        input_shape, committee_size = cls()._check_committee_results(predict_matrices)
        if len(input_shape) == 2:
            label_num = input_shape[1]
            # calc kl div for each instance
            for i in range(input_shape[0]):
                instance_mat = np.array([X[i, :] for X in predict_matrices if X is not None])
                tmp = 0
                # calc each label
                for lab in range(label_num):
                    committee_consensus = np.sum(instance_mat[:, lab]) / committee_size
                    for committee in range(committee_size):
                        tmp += instance_mat[committee, lab] * np.log(instance_mat[committee, lab] / committee_consensus)
                score.append(tmp)
        else:
            raise Exception(
                "A 2D probabilistic prediction matrix must be provided, with the shape like [n_samples, n_class]")
        return score


class QureyExpectedErrorReduction(BaseIndexQuery):
    """The Expected Error Reduction (ERR) algorithm.

    The idea is to estimate the expected future error of a model trained using label set and <x, y> on
    the remaining unlabeled instances in U (which is assumed to be representative of
    the test distribution, and used as a sort of validation set), and query the instance
    with minimal expected future error (sometimes called risk)

    This algorithm needs to re-train the model for multiple times.
    So There are 2 contraints to the given model.
    1. It is a sklearn model (or a model who implements their api).
    2. It has the probabilistic output function predict_proba.

    If your model does not meet the conditions.
    You can use the default logistic regression model to choose the instances
    by passing None to the model parameter.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    References
    ----------
    [1] N. Roy and A. McCallum. Toward optimal active learning through sampling
        estimation of error reduction. In Proceedings of the International Conference on
        Machine Learning (ICML), pages 441–448. Morgan Kaufmann, 2001.

    """

    def __init__(self, X=None, y=None):
        super(QureyExpectedErrorReduction, self).__init__(X, y)

    def log_loss(self, prob):
        """Compute expected log-loss.

        Parameters
        ----------
        prob: 2d array, shape [n_samples, n_classes]
            The probabilistic prediction matrix for the unlabeled set.

        Returns
        -------
        log_loss: float
            The sum of log_loss for the prob.
        """
        log_loss = 0.0
        for i in range(len(prob)):
            for p in list(prob[i]):
                log_loss -= p * np.log(p)
        return log_loss

    def select(self, label_index, unlabel_index, model=None, batch_size=1):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        # get unlabel_x
        if self.X is None or self.y is None:
            raise Exception('Data matrix is not provided, use select_by_prediction_mat() instead.')
        if model is None:
            model = LogisticRegression(solver='liblinear')
            model.fit(self.X[label_index if isinstance(label_index, (list, np.ndarray)) else label_index.index],
                      self.y[label_index if isinstance(label_index, (list, np.ndarray)) else label_index.index])

        unlabel_x = self.X[unlabel_index]
        label_y = self.y[label_index]
        ##################################

        classes = np.unique(self.y)
        pv, spv = _get_proba_pred(unlabel_x, model)
        scores = []
        for i in range(spv[0]):
            new_train_inds = np.append(label_index, unlabel_index[i])
            new_train_X = self.X[new_train_inds, :]
            unlabel_ind = list(unlabel_index)
            unlabel_ind.pop(i)
            new_unlabel_X = self.X[unlabel_ind, :]
            score = []
            for yi in classes:
                new_model = copy.deepcopy(model)
                new_model.fit(new_train_X, np.append(label_y, yi))
                prob = new_model.predict_proba(new_unlabel_X)
                score.append(pv[i, yi] * self.log_loss(prob))
            scores.append(np.sum(score))

        return unlabel_index[nsmallestarg(scores, batch_size)]


class QueryInstanceQUIRE(BaseIndexQuery):
    """Querying Informative and Representative Examples (QUIRE)

    Query the most informative and representative examples where the metrics
    measuring and combining are done using min-max approach. Note that, QUIRE is 
    not a batch mode active learning algorithm, it will select only one instance
    for querying at each iteration. Also, it does not need a model to evaluate the
    unlabeled data.

    The implementation refers to the project: https://github.com/ntucllab/libact

    NOTE: QUIRE is better to be used with RBF kernel, and usually the performance is
    good even without fine parameter tuning (that is, it is not very sensitive to
    parameter setting, and using default parameter setting is usually fine)

    Warning: QUIRE must NOT be used with linear kernel on non-textual data.

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    train_idx: array-like
        the index of training data.

    lambda: float, optional (default=1.0)
        A regularization parameter used in the regularization learning
        framework.

    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    References
    ----------
    [1] Yang, Y.-Y.; Lee, S.-C.; Chung, Y.-A.; Wu, T.-E.; Chen, S.-
        A.; and Lin, H.-T. 2017. libact: Pool-based active learning
        in python. Technical report, National Taiwan University.
        available as arXiv preprint https://arxiv.org/abs/
        1710.00379.

    [2] Huang, S.; Jin, R.; and Zhou, Z. 2014. Active learning by
        querying informative and representative examples. IEEE
        Transactions on Pattern Analysis and Machine Intelligence
        36(10):1936–1949
    """

    def __init__(self, X, y, train_idx, **kwargs):
        # K: kernel matrix
        #
        X = np.asarray(X)[train_idx]
        y = np.asarray(y)[train_idx]
        self._train_idx = np.asarray(train_idx)

        self.y = np.array(y)
        self.lmbda = kwargs.pop('lambda', 1.)
        self.kernel = kwargs.pop('kernel', 'rbf')
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=kwargs.pop('coef0', 1),
                                       degree=kwargs.pop('degree', 3),
                                       gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self.K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self.K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))
        self.L = np.linalg.inv(self.K + self.lmbda * np.eye(len(X)))

    def select(self, label_index, unlabel_index, **kwargs):
        """Select one instance from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        if len(unlabel_index) <= 1:
            return list(unlabel_index)
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)

        # build map from value to index
        label_index_in_train = [np.where(self._train_idx == i)[0][0] for i in label_index]
        unlabel_index_in_train = [np.where(self._train_idx == i)[0][0] for i in unlabel_index]
        # end

        L = self.L
        Lindex = list(label_index_in_train)
        Uindex = list(unlabel_index_in_train)
        query_index = -1
        min_eva = np.inf
        # y_labeled = np.array([label for label in self.y if label is not None])
        y_labeled = self.y[Lindex]
        det_Laa = np.linalg.det(L[np.ix_(Uindex, Uindex)])
        # efficient computation of inv(Laa)
        M3 = np.dot(self.K[np.ix_(Uindex, Lindex)],
                    np.linalg.inv(self.lmbda * np.eye(len(Lindex))))
        M2 = np.dot(M3, self.K[np.ix_(Lindex, Uindex)])
        M1 = self.lmbda * np.eye(len(Uindex)) + self.K[np.ix_(Uindex, Uindex)]
        inv_Laa = M1 - M2
        iList = list(range(len(Uindex)))
        if len(iList) == 1:
            return Uindex[0]
        for i, each_index in enumerate(Uindex):
            # go through all unlabeled instances and compute their evaluation
            # values one by one
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            iList_r = iList[:]
            iList_r.remove(i)
            inv_Luu = inv_Laa[np.ix_(iList_r, iList_r)] - 1 / inv_Laa[i, i] * \
                                                          np.dot(inv_Laa[iList_r, i], inv_Laa[iList_r, i].T)
            tmp = np.dot(
                L[each_index][Lindex] -
                np.dot(
                    np.dot(
                        L[each_index][Uindex_r],
                        inv_Luu
                    ),
                    L[np.ix_(Uindex_r, Lindex)]
                ),
                y_labeled,
            )
            eva = L[each_index][each_index] - \
                  det_Laa / L[each_index][each_index] + 2 * np.abs(tmp)

            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        return [self._train_idx[query_index]]


class QueryInstanceGraphDensity(BaseIndexQuery):
    """Diversity promoting sampling method that uses graph density to determine
    most representative points.

    The implementation refers to the https://github.com/google/active-learning

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    train_idx: array-like
        the index of training data.

    metric: str, optional (default='manhattan')
        the distance metric.
        valid metric = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                      'braycurtis', 'canberra', 'chebyshev', 'correlation',
                      'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                      'russellrao', 'seuclidean', 'sokalmichener',
                      'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]

    References
    ----------
    [1] Ebert, S.; Fritz, M.; and Schiele, B. 2012. RALF: A reinforced
        active learning formulation for object class recognition.
        In 2012 IEEE Conference on Computer Vision and  Pattern Recognition,
        Providence, RI, USA, June 16-21, 2012,
        3626–3633.
    """

    def __init__(self, X, y, train_idx, metric='manhattan'):
        self.metric = metric
        super(QueryInstanceGraphDensity, self).__init__(X, y)
        # Set gamma for gaussian kernel to be equal to 1/n_features
        self.gamma = 1. / self.X.shape[1]
        self.flat_X = X[train_idx, :]
        self.train_idx = train_idx
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._compute_graph_density()

    def _compute_graph_density(self, n_neighbor=10):
        # kneighbors graph is constructed using k=10
        connect = kneighbors_graph(self.flat_X, n_neighbor, p=1)
        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()
        inds = zip(neighbors[0], neighbors[1])
        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]
            distance = pairwise_distances(self.flat_X[[i]], self.flat_X[[j]], metric=self.metric)
            distance = distance[0, 0]
            weight = np.exp(-distance * self.gamma)
            connect[i, j] = weight
            connect[j, i] = weight
        self.connect = connect
        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.
        self.graph_density = np.zeros(self.flat_X.shape[0])
        for i in np.arange(self.flat_X.shape[0]):
            self.graph_density[i] = connect[i, :].sum() / (connect[i, :] > 0).sum()
        self.starting_density = copy.deepcopy(self.graph_density)

    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        # If a neighbor has already been sampled, reduce the graph density
        # for its direct neighbors to promote diversity.
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index
        batch = set()

        # build map from value to index
        try:
            label_index_in_train = [self.train_idx.index(i) for i in label_index]
        except:
            label_index_in_train = [np.where(self.train_idx == i)[0][0] for i in label_index]
        # end

        self.graph_density[label_index_in_train] = min(self.graph_density) - 1
        selected_arr = []
        while len(batch) < batch_size:
            selected = np.argmax(self.graph_density)
            selected_arr.append(selected)
            neighbors = (self.connect[selected, :] > 0).nonzero()[1]
            self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
            assert (self.train_idx[selected] in unlabel_index)
            batch.add(self.train_idx[selected])
            self.graph_density[label_index_in_train] = min(self.graph_density) - 1
            # self.graph_density[list(batch)] = min(self.graph_density) - 1
            self.graph_density[selected_arr] = min(self.graph_density) - 1
        return list(batch)

    def to_dict(self):
        """
        Return the connectivity and graph_density in the form of a dictionary.
        """
        output = {}
        output['connectivity'] = self.connect
        output['graph_density'] = self.starting_density
        return output


class QueryInstanceBMDR(BaseIndexQuery):
    """Discriminative and Representative Queries for Batch Mode Active Learning (BMDR)
    will query a batch of informative and representative examples by minimizing the ERM risk bound
    of active learning.

    This method needs to solve a quadratic programming problem for multiple times at one query which
    is time consuming in the relative large dataset (e.g., more than thousands of unlabeled examples).
    Note that, the solving speed is also influenced by kernel function. In our testing, the gaussian
    kernel takes more time to solve the problem.
    The QP solver is cvxpy here.

    The model used for instances selection is a linear regression model with the kernel form.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    beta: float, optional (default=1000)
        The MMD parameter.

    gamma: float, optional (default=0.1)
        The l2-norm regularizer parameter.

    rho: float, optional (default=1)
        The parameter used in ADMM.

    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma_ker : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    References
    ----------
    [1] Wang, Z., and Ye, J. 2013. Querying discriminative and
        representative samples for batch mode active learning. In The
        19th ACM SIGKDD International Conference on Knowledge
        Discovery and Data Mining, 158-166.
    """

    def __init__(self, X, y, beta=1000, gamma=0.1, rho=1, **kwargs):
        try:
            import cvxpy
            self._cvxpy = cvxpy
        except:
            raise ImportError("This method need cvxpy to solve the QP problem."
                              "Please refer to https://www.cvxpy.org/install/index.html "
                              "install cvxpy manually before using.")

        # K: kernel matrix
        super(QueryInstanceBMDR, self).__init__(X, y)
        ul = unique_labels(self.y)
        if len(ul) != 2:
            warnings.warn("This query strategy is implemented for binary classification only.",
                          category=FunctionWarning)
        if len(ul) == 2 and {1, -1} != set(ul):
            y_temp = np.array(copy.deepcopy(self.y))
            y_temp[y_temp == ul[0]] = 1
            y_temp[y_temp == ul[1]] = -1
            self.y = y_temp

        self._beta = beta
        self._gamma = gamma
        self._rho = rho

        # calc kernel
        self._kernel = kwargs.pop('kernel', 'rbf')
        if self._kernel == 'rbf':
            self._K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma_ker', 1.))
        elif self._kernel == 'poly':
            self._K = polynomial_kernel(X=X,
                                        Y=X,
                                        coef0=kwargs.pop('coef0', 1),
                                        degree=kwargs.pop('degree', 3),
                                        gamma=kwargs.pop('gamma_ker', 1.))
        elif self._kernel == 'linear':
            self._K = linear_kernel(X=X, Y=X)
        elif hasattr(self._kernel, '__call__'):
            self._K = self._kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self._K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self._K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))

    def __getstate__(self):
        pickle_seq = (
            self.X,
            self.y,
            self._beta,
            self._gamma,
            self._rho,
            self._kernel,
            self._K
        )
        return pickle_seq

    def __setstate__(self, state):
        self.X, self.y, self._beta, self._gamma, self._rho, self._kernel, self._K = state
        import cvxpy
        self._cvxpy = cvxpy

    def select(self, label_index, unlabel_index, batch_size=5, qp_solver='ECOS', **kwargs):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.

        qp_solver: str, optional (default='ECOS')
            The solver in cvxpy to solve QP, must be one of
            ['ECOS', 'OSQP']
            ECOS: https://www.embotech.com/ECOS
            OSQP: https://osqp.org/

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        cvxpy = self._cvxpy
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        KLL = self._K[np.ix_(label_index, label_index)]
        KLU = self._K[np.ix_(label_index, unlabel_index)]
        KUU = self._K[np.ix_(unlabel_index, unlabel_index)]

        L_len = len(label_index)
        U_len = len(unlabel_index)
        N = L_len + U_len

        # precision of ADMM
        MAX_ITER = 1000
        ABSTOL = 1e-4
        RELTOL = 1e-2

        # train a linear model in kernel form for
        tau = np.linalg.inv(KLL + self._gamma * np.eye(L_len)).dot(self.y[label_index])

        # start optimization
        last_round_selected = []
        iter_round = 0
        while 1:
            iter_round += 1
            # solve QP
            P = 0.5 * self._beta * KUU
            pred_of_unlab = tau.dot(KLU)
            a = pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab)
            q = self._beta * (
                (U_len - batch_size) / N * np.ones(L_len).dot(KLU) - (L_len + batch_size) / N * np.ones(U_len).dot(
                    KUU)) + a

            # cvx
            x = cvxpy.Variable(U_len)
            objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
            constraints = [0 <= x, x <= 1, sum(x) == batch_size]
            prob = cvxpy.Problem(objective, constraints)
            # The optimal objective value is returned by `prob.solve()`.
            # print(prob.is_qp())
            result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
            # The optimal value for x is stored in `x.value`.
            # print(x.value)
            dr_weight = np.array(x.value)
            if len(np.shape(dr_weight)) == 2:
                dr_weight = dr_weight.T[0]
            # end cvx

            # record selected indexes and judge convergence
            dr_largest = nlargestarg(dr_weight, batch_size)
            select_ind = np.asarray(unlabel_index)[dr_largest]
            if set(last_round_selected) == set(select_ind) or iter_round > 15:
                return select_ind
            else:
                last_round_selected = copy.copy(select_ind)
            # print(dr_weight[dr_largest])

            # ADMM optimization process
            delta = np.zeros(batch_size)  # dual variable in ADMM
            KLQ = self._K[np.ix_(label_index, select_ind)]
            z = tau.dot(KLQ)

            for solver_iter in range(MAX_ITER):
                # tau update
                A = KLL.dot(KLL) + self._rho / 2 * KLQ.dot(KLQ.T) + self._gamma * KLL
                r = self.y[label_index].dot(KLL) + 0.5 * delta.dot(KLQ.T) + self._rho / 2 * z.dot(KLQ.T)
                tau = np.linalg.pinv(A).dot(r)

                # z update
                zold = z
                v = (self._rho * tau.dot(KLQ) - delta) / (self._rho + 2)
                ita = 2 / (self._rho + 2)
                z_sign = np.sign(v)
                z_sign[z_sign == 0] = 1
                ztp = (np.abs(v) - ita * np.ones(len(v)))
                ztp[ztp < 0] = 0
                z = z_sign * ztp

                # delta update
                delta += self._rho * (z - tau.dot(KLQ))

                # judge convergence
                r_norm = np.linalg.norm((tau.dot(KLQ) - z))
                s_norm = np.linalg.norm(-self._rho * (z - zold))
                eps_pri = np.sqrt(batch_size) * ABSTOL + RELTOL * max(np.linalg.norm(z), np.linalg.norm(tau.dot(KLQ)))
                eps_dual = np.sqrt(batch_size) * ABSTOL + RELTOL * np.linalg.norm(delta)
                if r_norm < eps_pri and s_norm < eps_dual:
                    break


class QueryInstanceSPAL(BaseIndexQuery):
    """Self-Paced Active Learning: Query the Right Thing at the Right Time (SPAL)
    will query a batch of informative, representative and easy examples by minimizing a
    well designed objective function.

    The QP solver is cvxpy here.

    The model used for instances selection is a linear regression model with the kernel form.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    mu: float, optional (default=0.1)
        The MMD parameter.

    gamma: float, optional (default=0.1)
        The l2-norm regularizer parameter.

    rho: float, optional (default=1)
        The parameter used in ADMM.

    lambda_init: float, optional (default=0.1)
        The initial value of lambda used in SP regularizer.

    lambda_pace: float, optional (default=0.01)
        The pace of lambda when updating.

    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma_ker : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    References
    ----------
    [1] Wang, Z., and Ye, J. 2013. Querying discriminative and
        representative samples for batch mode active learning. In The
        19th ACM SIGKDD International Conference on Knowledge
        Discovery and Data Mining, 158-166.
    """

    def __init__(self, X, y, mu=0.1, gamma=0.1, rho=1, lambda_init=0.1, lambda_pace=0.01, **kwargs):
        try:
            import cvxpy
            self._cvxpy = cvxpy
        except:
            raise ImportError("This method need cvxpy to solve the QP problem."
                              "Please refer to https://www.cvxpy.org/install/index.html "
                              "install cvxpy manually before using.")

        # K: kernel matrix
        super(QueryInstanceSPAL, self).__init__(X, y)
        ul = unique_labels(self.y)
        if len(unique_labels(self.y)) != 2:
            warnings.warn("This query strategy is implemented for binary classification only.",
                          category=FunctionWarning)
        if len(ul) == 2 and {1, -1} != set(ul):
            y_temp = np.array(copy.deepcopy(self.y))
            y_temp[y_temp == ul[0]] = 1
            y_temp[y_temp == ul[1]] = -1
            self.y = y_temp

        self._mu = mu
        self._gamma = gamma
        self._rho = rho
        self._lambda_init = lambda_init
        self._lambda_pace = lambda_pace
        self._lambda = lambda_init

        # calc kernel
        self._kernel = kwargs.pop('kernel', 'rbf')
        if self._kernel == 'rbf':
            self._K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma_ker', 1.))
        elif self._kernel == 'poly':
            self._K = polynomial_kernel(X=X,
                                        Y=X,
                                        coef0=kwargs.pop('coef0', 1),
                                        degree=kwargs.pop('degree', 3),
                                        gamma=kwargs.pop('gamma_ker', 1.))
        elif self._kernel == 'linear':
            self._K = linear_kernel(X=X, Y=X)
        elif hasattr(self._kernel, '__call__'):
            self._K = self._kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        if not isinstance(self._K, np.ndarray):
            raise TypeError('K should be an ndarray')
        if self._K.shape != (len(X), len(X)):
            raise ValueError(
                'kernel should have size (%d, %d)' % (len(X), len(X)))

    def __getstate__(self):
        pickle_seq = (
            self.X,
            self.y,
            self._mu,
            self._gamma,
            self._rho,
            self._lambda,
            self._lambda_init,
            self._lambda_pace,
            self._kernel,
            self._K
        )
        return pickle_seq

    def __setstate__(self, state):
        self.X, self.y, self._mu, self._gamma, self._rho, self._lambda, self._lambda_init, self._lambda_pace, self._kernel, self._K = state
        import cvxpy
        self._cvxpy = cvxpy

    def select(self, label_index, unlabel_index, batch_size=5, qp_solver='ECOS', **kwargs):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.

        qp_solver: str, optional (default='ECOS')
            The solver in cvxpy to solve QP, must be one of
            ['ECOS', 'OSQP']
            ECOS: https://www.embotech.com/ECOS
            OSQP: https://osqp.org/

        Returns
        -------
        selected_idx: list
            The selected indexes which is a subset of unlabel_index.
        """
        cvxpy = self._cvxpy
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        KLL = self._K[np.ix_(label_index, label_index)]
        KLU = self._K[np.ix_(label_index, unlabel_index)]
        KUU = self._K[np.ix_(unlabel_index, unlabel_index)]

        L_len = len(label_index)
        U_len = len(unlabel_index)
        N = L_len + U_len

        # precision of ADMM
        MAX_ITER = 1000
        ABSTOL = 1e-4
        RELTOL = 1e-2

        # train a linear model in kernel form for
        theta = np.linalg.inv(KLL + self._gamma * np.eye(L_len)).dot(self.y[label_index])

        # start optimization
        dr_weight = np.ones(U_len)  # informativeness % representativeness
        es_weight = np.ones(U_len)  # easiness
        last_round_selected = []
        iter_round = 0
        while 1:
            iter_round += 1
            # solve QP
            P = 0.5 * self._mu * KUU
            pred_of_unlab = theta.dot(KLU)
            a = es_weight * (pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab))
            q = self._mu * (
                (U_len - batch_size) / N * np.ones(L_len).dot(KLU) - (L_len + batch_size) / N * np.ones(U_len).dot(
                    KUU)) + a
            # cvx
            x = cvxpy.Variable(U_len)
            objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
            constraints = [0 <= x, x <= 1, es_weight * x == batch_size]
            prob = cvxpy.Problem(objective, constraints)
            # The optimal objective value is returned by `prob.solve()`.
            result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)
            # Sometimes the constraints can not be satisfied,
            # thus we relax the constraints to get an approximate solution.
            if not (type(result) == float and result != float('inf') and result != float('-inf')):
                P = 0.5 * self._mu * KUU
                pred_of_unlab = theta.dot(KLU)
                a = es_weight * (pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab))
                q = self._mu * (
                    (U_len - batch_size) / N * np.ones(L_len).dot(KLU) - (L_len + batch_size) / N * np.ones(U_len).dot(
                        KUU)) + a
                # cvx
                x = cvxpy.Variable(U_len)
                objective = cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T * x)
                constraints = [0 <= x, x <= 1]
                prob = cvxpy.Problem(objective, constraints)
                # The optimal objective value is returned by `prob.solve()`.
                result = prob.solve(solver=cvxpy.OSQP if qp_solver == 'OSQP' else cvxpy.ECOS)

            # The optimal value for x is stored in `x.value`.
            # print(x.value)
            dr_weight = np.array(x.value)
            # print(dr_weight)
            # print(result)
            if len(np.shape(dr_weight)) == 2:
                dr_weight = dr_weight.T[0]
            # end cvx

            # update easiness weight
            worst_loss = dr_weight * (pred_of_unlab * pred_of_unlab + 2 * np.abs(pred_of_unlab))
            es_weight = np.zeros(U_len)
            es_weight_tmp = 1 - (worst_loss / self._lambda)
            update_indices = np.nonzero(worst_loss < self._lambda)[0]
            es_weight[update_indices] = es_weight_tmp[update_indices]

            # record selected indexes and judge convergence
            dr_largest = nlargestarg(dr_weight * es_weight, batch_size)
            select_ind = np.asarray(unlabel_index)[dr_largest]
            if set(last_round_selected) == set(select_ind) or iter_round > 15:
                return select_ind
            else:
                last_round_selected = copy.copy(select_ind)
                # print(dr_largest)
                # print(dr_weight[dr_largest])

            # ADMM optimization process
            # Filter less important instances for efficiency
            mix_weight = dr_weight * es_weight
            mix_weight[mix_weight < 0] = 0
            validind = np.nonzero(mix_weight > 0.001)[0]
            if len(validind) < 1:
                validind = nlargestarg(mix_weight, 1)
            vKlu = KLU[:, validind]

            delta = np.zeros(len(validind))  # dual variable in ADMM
            z = theta.dot(vKlu)  # auxiliary variable in ADMM

            # pre-computed constants in ADMM
            A = 2 * KLL.dot(KLL) + self._rho * vKlu.dot(vKlu.T) + 2 * self._gamma * KLL
            pinvA = np.linalg.pinv(A)
            rz = self._rho * vKlu
            rc = 2 * KLL.dot(self.y[label_index])
            kdenom = np.sqrt(mix_weight[validind] + self._rho / 2)
            ci = mix_weight[validind] / kdenom

            for solver_iter in range(MAX_ITER):
                # theta update
                r = rz.dot(z.T) + vKlu.dot(delta) + rc
                theta = pinvA.dot(r)

                # z update
                zold = z
                vud = self._rho * theta.dot(vKlu)
                vi = (vud - delta) / (2 * kdenom)
                ztmp = np.abs(vi) - ci
                ztmp[ztmp < 0] = 0
                ksi = np.sign(vi) * ztmp
                z = ksi / kdenom

                # delta update
                delta += self._rho * (z - theta.dot(vKlu))

                # judge convergence
                r_norm = np.linalg.norm((theta.dot(vKlu) - z))
                s_norm = np.linalg.norm(-self._rho * (z - zold))
                eps_pri = np.sqrt(len(validind)) * ABSTOL + RELTOL * max(np.linalg.norm(z),
                                                                         np.linalg.norm(theta.dot(vKlu)))
                eps_dual = np.sqrt(len(validind)) * ABSTOL + RELTOL * np.linalg.norm(delta)
                if r_norm < eps_pri and s_norm < eps_dual:
                    break


class QueryInstanceLAL(BaseIndexQuery):
    """The key idea of LAL is to train a regressor that predicts the
    expected error reduction for a candidate sample in a particular learning state.

    The regressor is trained on 2D datasets and can score unseen data from real
    datasets. The method yields strategies that work well on real data from a
    wide range of domains.

    In alipy, LAL will use a pre-extracted data provided by the authors to train
    the regressor. It will download the data file if no accepted file is found.
    You can also download 'LAL-iterativetree-simulatedunbalanced-big.npz'
    and 'LAL-randomtree-simulatedunbalanced-big.npz' from https://github.com/ksenia-konyushkova/LAL.
    and specify the dir to the file for training.

    The implementation is refer to the https://github.com/ksenia-konyushkova/LAL/ directly.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    mode: str, optional (default='LAL_iterative')
        The mode of data sampling. must be one of 'LAL_iterative', 'LAL_independent'.

    data_path: str, optional (default='.')
        Path to store the data file for training.
        The path should be a dir, and the file name should be
        'LAL-iterativetree-simulatedunbalanced-big.npz' or 'LAL-randomtree-simulatedunbalanced-big.npz'.
        If no accepted files are detected, it will download the pre-extracted data file to the given path.

    cls_est: int, optional (default=50)
        The number of estimator used for training the random forest whose role
        is calculating the features for selector.

    train_slt: bool, optional (default=True)
        Whether to train a selector in initializing.

    References
    ----------
    [1] Ksenia Konyushkova, and Sznitman Raphael. 2017. Learning
        Active Learning from Data. In The 31st Conference on
        Neural Information Processing Systems (NIPS 2017), 4228-4238.

    """

    def __init__(self, X, y, mode='LAL_iterative', data_path='.', cls_est=50, train_slt=True, **kwargs):
        super(QueryInstanceLAL, self).__init__(X, y)
        if len(unique_labels(self.y)) != 2:
            warnings.warn("This query strategy is implemented for binary classification only.",
                          category=FunctionWarning)
        if not os.path.isdir(data_path):
            raise ValueError("Please pass the directory of the file.")
        self._iter_path = os.path.join(data_path, 'LAL-iterativetree-simulatedunbalanced-big.npz')
        self._rand_path = os.path.join(data_path, 'LAL-randomtree-simulatedunbalanced-big.npz')
        assert mode in ['LAL_iterative', 'LAL_independent']
        self._mode = mode
        self._selector = None
        self.model = RandomForestClassifier(n_estimators=cls_est, oob_score=True, n_jobs=8)
        if train_slt:
            self.download_data()
            self.train_selector_from_file()

    def download_data(self):
        iter_url = 'https://raw.githubusercontent.com/ksenia-konyushkova/LAL/master/lal%20datasets/LAL-iterativetree-simulatedunbalanced-big.npz'
        rand_url = 'https://raw.githubusercontent.com/ksenia-konyushkova/LAL/master/lal%20datasets/LAL-randomtree-simulatedunbalanced-big.npz'
        chunk_size = 64 * 1024
        if self._mode == 'LAL_iterative':
            if not os.path.exists(self._iter_path):
                # download file
                print(str(self._iter_path) + " file is not found. Starting to download...")
                import requests
                f = requests.get(iter_url, stream=True)
                total_size = f.headers['content-length']
                download_count = 0
                with open(self._iter_path, "wb") as code:
                    for chunk in f.iter_content(chunk_size=chunk_size):
                        download_count += 1
                        if chunk:
                            print("\rProgress:%.2f%% (%d/%d)\t - %s" % (
                                min(download_count * chunk_size * 100 / float(total_size), 100.00),
                                min(download_count * chunk_size, int(total_size)),
                                int(total_size), iter_url), end='')
                            code.write(chunk)
                print('\nDownload end.')
            return self._iter_path
            # self.train_selector_from_file(self._iter_path)
        else:
            if not os.path.exists(self._rand_path):
                # download file
                print(str(self._rand_path) + " file is not found. Starting to download...")
                import requests
                f = requests.get(rand_url, stream=True)
                total_size = f.headers['content-length']
                download_count = 0
                with open(self._rand_path, "wb") as code:
                    for chunk in f.iter_content(chunk_size=chunk_size):
                        download_count += 1
                        if chunk:
                            print("\rProgress:%.2f%% (%d/%d) - %s" % (
                                min(download_count * chunk_size * 100 / float(total_size), 100.00),
                                min(download_count * chunk_size, int(total_size)),
                                int(total_size), rand_url), end='')
                            code.write(chunk)
                print('\nDownload end.')
            return self._rand_path
            # self.train_selector_from_file(self._rand_path)

    def train_selector_from_file(self, file_path=None, reg_est=2000, reg_depth=40, feat=6):
        """Train a random forest as the instance selector.
        Note that, if the parameters of the forest is too
        high to your computer, it will take a lot of time
        for training.

        Parameters
        ----------
        file_path: str, optional (default=None)
            The path to the specific data file.

        reg_est: int, optional (default=2000)
            The number of estimators of the forest.

        reg_depth: int, optional (default=40)
            The depth of the forest.

        feat: int, optional (default=6)
            The feat of the forest.
        """

        if file_path is None:
            file_path = self._iter_path if self._mode == 'LAL_iterative' else self._rand_path
        else:
            if os.path.isdir(file_path):
                raise ValueError("Please pass the path to a specific file, not a directory.")
        parameters = {'est': reg_est, 'depth': reg_depth, 'feat': feat}
        regression_data = np.load(file_path)
        regression_features = regression_data['arr_0']
        regression_labels = regression_data['arr_1']

        print('Building lal regression model from ' + file_path)
        lalModel1 = RandomForestRegressor(n_estimators=parameters['est'], max_depth=parameters['depth'],
                                          max_features=parameters['feat'], oob_score=True, n_jobs=8)

        lalModel1.fit(regression_features, np.ravel(regression_labels))

        print('Done!')
        print('Oob score = ', lalModel1.oob_score_)
        self._selector = lalModel1

    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        if self._selector is None:
            raise ValueError(
                "Please train the selection regressor first.\nUse train_selector_from_file(path) "
                "if you have already download the data for training. Otherwise, use download_data() "
                "method to download the data first.")
        assert (batch_size > 0)
        assert (isinstance(unlabel_index, collections.Iterable))
        assert (isinstance(label_index, collections.Iterable))
        unlabel_index = np.asarray(unlabel_index)
        label_index = np.asarray(label_index)
        if len(unlabel_index) <= batch_size:
            return unlabel_index

        unknown_data = self.X[unlabel_index]
        known_labels = self.y[label_index]
        known_data = self.X[label_index]
        n_lablled = len(label_index)
        n_dim = self.X.shape[1]
        self.model.fit(known_data, known_labels)

        # predictions of the trees
        temp = np.array([tree.predict_proba(unknown_data)[:, 0] for tree in self.model.estimators_])
        # - average and standard deviation of the predicted scores
        f_1 = np.mean(temp, axis=0)
        f_2 = np.std(temp, axis=0)
        # - proportion of positive points
        f_3 = (sum(known_labels > 0) / n_lablled) * np.ones_like(f_1)
        # the score estimated on out of bag estimate
        f_4 = self.model.oob_score_ * np.ones_like(f_1)
        # - coeficient of variance of feature importance
        f_5 = np.std(self.model.feature_importances_ / n_dim) * np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(f_2, axis=0) * np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in self.model.estimators_])) * np.ones_like(f_1)
        # - number of already labelled datapoints
        f_8 = n_lablled * np.ones_like(f_1)

        # all the featrues put together for regressor
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        LALfeatures = np.transpose(LALfeatures)
        # predict the expercted reduction in the error by adding the point
        LALprediction = self._selector.predict(LALfeatures)
        # select the datapoint with the biggest reduction in the error
        selectedIndex1toN = nlargestarg(LALprediction, batch_size)
        # print(LALprediction)
        # print(np.asarray(LALprediction)[selectedIndex1toN])
        # retrieve the real index of the selected datapoint
        selectedIndex = np.asarray(unlabel_index)[selectedIndex1toN]
        return selectedIndex
