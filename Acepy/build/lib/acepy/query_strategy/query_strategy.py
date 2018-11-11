"""
Pre-defined classical query strategy.

References:
[1] Settles, B. 2009. Active learning literature survey. Technical
    report, University of Wisconsin-Madison.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import collections
import copy
import warnings

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

import acepy.utils.interface
from acepy.utils.misc import nsmallestarg, randperm, nlargestarg


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


class QueryInstanceUncertainty(acepy.utils.interface.BaseIndexQuery):
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
            model = LogisticRegression()
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

        predict: 2d array, shape [n_samples, n_classes] or [n_samples]
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


class QueryRandom(acepy.utils.interface.BaseQueryStrategy):
    """Randomly sample a batch of indexes from the unlabel indexes."""

    def select(self, unlabel_index, batch_size=1):
        """Select indexes randomly.

        Parameters
        ----------
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


class QueryInstanceQBC(acepy.utils.interface.BaseIndexQuery):
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
            model = LogisticRegression()
            model.fit(self.X[label_index if isinstance(label_index, (list, np.ndarray)) else label_index.index],
                      self.y[label_index if isinstance(label_index, (list, np.ndarray)) else label_index.index])

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


class QureyExpectedErrorReduction(acepy.utils.interface.BaseIndexQuery):
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
            model = LogisticRegression()
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
