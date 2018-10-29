"""
Pre-defined Performance
Implement classical methods
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from acepy.utils.tools import check_one_to_one_correspondence
 
 
__all__ = [
    'accuracy_score',
    'auc',
    'get_fps_tps_thresholds',
    'hamming_loss',
    'one_error',
    'coverage_error',
    'label_ranking_loss',
    'label_ranking_average_precision_score'
]


def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """
        Check that all arrays have consistent first dimensions.
    """
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def type_of_target(y):
    """Determine the type of data indicated by the target.
    """
    y = np.asarray(y)
    if(len(np.unique(y)) <= 2):
        if(y.ndim >= 2 and len(y[0]) > 1):
            return 'multilabel'
        else:
            return 'binary'
    elif(len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return 'multiclass'
    return 'unknown'
        

def _check_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d, while multilabel formats are returned
    as CSR sparse label indicators.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``

    y_true : array or indicator matrix

    y_pred : array or indicator matrix
    """
    check_consistent_length(y_true, y_pred)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    y_type = set([type_true, type_pred])
    if y_type == set(["binary", "multiclass"]):
        y_type = set(["multiclass"])

    if len(y_type) > 1:
        raise ValueError("Classification metrics can't handle a mix of {0} "
                         "and {1} targets".format(type_true, type_pred))

    # We can't have more than one value on y_type => The set is no more needed
    y_type = y_type.pop()

    # No metrics support "multiclass-multioutput" format
    if (y_type not in ["binary", "multiclass", "multilabel"]):
        raise ValueError("{0} is not supported".format(y_type))

    if y_type in ["binary", "multiclass"]:
        #Ravel column or 1d numpy array
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        if y_type == "binary":
            unique_values = np.union1d(y_true, y_pred)
            if len(unique_values) > 2:
                y_type = "multiclass"

    if y_type.startswith('multilabel'):
        y_true = csr_matrix(y_true)
        y_pred = csr_matrix(y_pred)
        y_type = 'multilabel'

    return y_type, y_true, y_pred


def accuracy_score(y_true, y_pred, sample_weight=None):
    """Accuracy classification score.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) _labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted _labels, as returned by a classifier.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float
    """
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)

    if y_type.startswith('multilabel'):
    
        differing_labels = np.diff((y_true - y_pred).indptr)
        score = differing_labels == 0
    else:
        score = y_true == y_pred
    return np.average(score, weights=sample_weight)


def zero_one_loss(y_true, y_pred, normalize=True, sample_weight=None):
    """Zero-one classification loss.

    If normalize is ``True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int). The best
    performance is 0.

    Read more in the :ref:`User Guide <zero_one_loss>`.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of misclassifications.
        Otherwise, return the fraction of misclassifications.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    loss : float or int,
        If ``normalize == True``, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int).
    """
    score = accuracy_score(y_true, y_pred,sample_weight=sample_weight)

    if normalize:
        return 1 - score
    else:
        if sample_weight is not None:
            n_samples = np.sum(sample_weight)
        else:
            n_samples = _num_samples(y_true)
        return n_samples - score


def f1_score(y_true, y_pred, pos_label=1, sample_weight=None):

    p, r, t = precision_recall_curve(y_true, y_pred, pos_label=pos_label,
                           sample_weight=sample_weight)
    
    return 2 * (p * r) / (p + r)


def auc(x, y, reorder=True):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    Parameters
    ----------
    x : array, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array, shape = [n]
        y coordinates.
    reorder : boolean, optional (default='deprecated')
        Whether to sort x before computing. If False, assume that x must be
        either monotonic increasing or monotonic decreasing. If True, y is
        used to break ties when sorting x. Make sure that y has a monotonic
        relation to x when setting reorder to True.

    Returns
    -------
    auc : float

    """
    check_consistent_length(x, y)

    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    if reorder is True:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("x is neither increasing nor decreasing "
                                 ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area


def get_fps_tps_thresholds(y_true, y_score, pos_label=None):
    '''
    Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    '''
    check_consistent_length(y_true, y_score)
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    thresholds = np.array(y_score)[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices]
    y_score = np.array(y_score)[desc_score_indices]
    tps = []
    fps = [] 
    for threshold in thresholds:
        y_prob = [1 if i >= threshold else 0 for i in y_score]
        result = [i == j for i, j in zip(y_true, y_prob)]
        postive = [i == 1 for i in y_prob]
        tp = [i and j for i, j in zip(result, postive)]
        fp = [(not i) and j for i, j in zip(result, postive)]
        tps.append(tp.count(True))
        fps.append(fp.count(True))
    return np.array(fps), np.array(tps), thresholds


def precision_recall_curve(y_true, y_score, pos_label=None,
                           sample_weight=None):
    '''
    Compute precision-recall pairs for different probability thresholds

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification in range {-1, 1} or {0, 1}.

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function.

    pos_label : int or str, default=None
        The label of the positive class

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    '''
    fps, tps, thresholds = get_fps_tps_thresholds(y_true, y_score, pos_label=pos_label)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def roc_curve(y_true, y_score, pos_label=None, sample_weight=None):
    '''Compute Receiver operating characteristic (ROC)

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    pos_label : int or str, default=None
        Label considered as positive and others are considered negative.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr. `thresholds[0]` represents no instances being predicted
        and is arbitrarily set to `max(y_score) + 1`.
    '''
    fps, tps, thresholds = get_fps_tps_thresholds(y_true, y_score, pos_label=pos_label)

    if np.array(tps).size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        raise ValueError("No negative samples in y_true,false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        raise ValueError("No positive samples in y_true,true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def roc_auc_score(y_true, y_score, pos_label=None, sample_weight=None):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary _labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers). For binary
        y_true, y_score is supposed to be the score of the class with greater
        label.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    auc : float
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label, sample_weight=None)
    return auc(fpr, tpr)    


def hamming_loss(y_true, y_pred):
    """Compute the average Hamming loss.

    """
    y_type, _, _ = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_type.startswith('multilabel'):
        num_samples, num_classses = np.array(y_true).shape
        n_differences = np.sum(y_true != y_pred)
        return n_differences / (num_samples * num_classses)
    elif y_type in ["binary", "multiclass"]:
        return np.sum(y_true != y_pred) / y_true.shape[0]
    else:
        raise ValueError("{0} is not supported".format(y_type))


def one_error(y_true, y_pred, sample_weight=None):
    '''

    '''
    check_consistent_length(y_true, y_pred, sample_weight)
    y_type = type_of_target(y_true)
    n_samples, n_labels = y_true.shape
    if y_type != "multilabel":
        raise ValueError("{0} format is not supported".format(y_type))
    n_differents = np.sum((y_true - y_pred), axis=1) == 0

    return n_differents.sum() / n_samples


def coverage_error(y_true, y_score, sample_weight=None):
    """Coverage error measure
    """
    check_consistent_length(y_true, y_score, sample_weight)
    y_type = type_of_target(y_true)
    if y_type != "multilabel":
        raise ValueError("{0} format is not supported".format(y_type))
    y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))
    y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))
    coverage = (y_score >= y_min_relevant).sum(axis=1)
    coverage = coverage.filled(0)
    return np.average(coverage, weights=sample_weight)


def label_ranking_loss(y_true, y_score, sample_weight=None):
    """Compute Ranking loss measure

    """
    check_consistent_length(y_true, y_score, sample_weight)

    y_type = type_of_target(y_true)
    if y_type not in ("multilabel",):
        raise ValueError("{0} format is not supported".format(y_type))

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    n_samples, n_labels = y_true.shape

    y_true = csr_matrix(y_true)

    loss = np.zeros(n_samples)
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        # Sort and bin the label scores
        unique_scores, unique_inverse = np.unique(y_score[i],
                                                  return_inverse=True)
        true_at_reversed_rank = np.bincount(
            unique_inverse[y_true.indices[start:stop]],
            minlength=len(unique_scores))
        all_at_reversed_rank = np.bincount(unique_inverse,
                                        minlength=len(unique_scores))
        false_at_reversed_rank = all_at_reversed_rank - true_at_reversed_rank

        # if the scores are ordered, it's possible to count the number of
        # incorrectly ordered paires in linear time by cumulatively counting
        # how many false _labels of a given score have a score higher than the
        # accumulated true _labels with lower score.
        loss[i] = np.dot(true_at_reversed_rank.cumsum(),
                         false_at_reversed_rank)

    n_positives = np.diff(y_true.indptr)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss /= ((n_labels - n_positives) * n_positives)

    # When there is no positive or no negative _labels, those values should
    # be consider as correct, i.e. the ranking doesn't matter.
    loss[np.logical_or(n_positives == 0, n_positives == n_labels)] = 0.

    return np.average(loss, weights=sample_weight)


def label_ranking_average_precision_score(y_true, y_score, sample_weight=None):
    """Compute ranking-based average precision

    Parameters
    ----------
    y_true : array or sparse matrix, shape = [n_samples, n_labels]
        True binary labels in binary indicator format.

    y_score : array, shape = [n_samples, n_labels]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : float

    """
    check_consistent_length(y_true, y_score, sample_weight)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    # Handle badly formatted array and the degenerate case with one label
    y_type = type_of_target(y_true)
    if (y_type != "multilabel" and
            not (y_type == "binary" and y_true.ndim == 2)):
        raise ValueError("{0} format is not supported".format(y_type))

    y_true = csr_matrix(y_true)
    y_score = -y_score

    n_samples, n_labels = y_true.shape

    out = 0.
    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):
        relevant = y_true.indices[start:stop]

        if (relevant.size == 0 or relevant.size == n_labels):
            # If all labels are relevant or unrelevant, the score is also
            # equal to 1. The label ranking has no meaning.
            out += 1.
            continue

        scores_i = y_score[i]
        rank = rankdata(scores_i, 'max')[relevant]
        L = rankdata(scores_i[relevant], 'max')
        aux = (L / rank).mean()
        if sample_weight is not None:
            aux = aux * sample_weight[i]
        out += aux

    if sample_weight is None:
        out /= n_samples
    else:
        out /= np.sum(sample_weight)

    return out


def find(instance, label1, label2):
    index1 = []
    index2 = []
    for i in range(instance.shape[0]):
        if instance[i] == label1:
            index1.append(i)
        if instance[i] == label2:
            index2.append(i)
    return index1, index2


def findmax(outputs):
    Max = -float("inf")    
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index


def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j        
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = float("inf")
    return sortX, index


def findIndex(a, b):
    for i in range(len(b)):
        if a == b[i]:
            return i

   
def micro_auc_score(y_true, y_score, sample_weight=None):
    check_consistent_length(y_true, y_score, sample_weight)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    # Handle badly formatted array and the degenerate case with one label
    y_type = type_of_target(y_true)
    if (y_type != "multilabel" and
            not (y_type == "binary" and y_true.ndim == 2)):
        raise ValueError("{0} format is not supported".format(y_type))

    test_data_num = y_score.shape[0]
    class_num = y_score.shape[1]
    P = []
    N = []
    labels_size = []
    not_labels_size = []
    AUC = 0
    for i in range(class_num):
        P.append([])
        N.append([])
    
    for i in range(test_data_num):
            for j in range(class_num):
                if y_true[i][j] == 1:
                    P[j].append(i)
                else:
                    N[j].append(i)
    
    for i in range(class_num):
        labels_size.append(len(P[i]))
        not_labels_size.append(len(N[i]))
    
    for i in range(class_num):
        auc = 0
        for j in range(labels_size[i]):
            for k in range(not_labels_size[i]):
                pos = y_score[P[i][j]][i]
                neg = y_score[N[i][k]][i]
                if pos > neg:
                    auc = auc + 1
        AUC = AUC + auc*1.0/(labels_size[i]*not_labels_size[i])
    return AUC*1.0/class_num


def average_precision_score(y_true, y_score, sample_weight=None):
    check_consistent_length(y_true, y_score, sample_weight)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shape")

    # Handle badly formatted array and the degenerate case with one label
    y_type = type_of_target(y_true)
    if (y_type != "multilabel" and
            not (y_type == "binary" and y_true.ndim == 2)):
        raise ValueError("{0} format is not supported".format(y_type))

    test_data_num = y_score.shape[0]
    class_num = y_score.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(y_true[i]) != class_num and sum(y_true[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(y_score[i])
            temp_test_target.append(y_true[i])
            labels_size.append(sum(y_true[i] == 1))
            index1, index2 = find(y_true[i], 1, 0)            
            labels_index.append(index1)
            not_labels_index.append(index2)
    
    aveprec = 0
    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num,))     
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            summary = summary + sum(indicator[loc:class_num])*1.0/(class_num-loc)
        aveprec = aveprec + summary*1.0/labels_size[i]
    return aveprec*1.0/test_data_num


if __name__ == '__main__':
    # print(roc_auc_score(y, scores))
    # print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))
    # fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)

    # print('fpr is ', fpr)
    # print('tpr is ', tpr)
    # y_true = np.array([[1, 0, 1, 0],[0, 1, 0, 1],[1, 0, 0, 1],[0, 1, 1, 0],[1, 0, 0, 0]])
    # y_socre = np.array([[0.9, 0.0, 0.4, 0.6],[0.1, 0.8, 0.0, 0.8],[0.8, 0.0, 0.1, 0.7],[0.1, 0.7, 0.1, 0.2],[1.0, 0, 0, 1.0]])
    # y_true = np.array([[1, 0, 1, 0],[0, 1, 0, 1]])
    # y_socre = np.array([[0.9, 0.0, 0.4, 0.6],[0.1, 0.8, 0.0, 0.8]])

    # print(label_ranking_average_precision_score(y_true,y_socre))
    # y_true = np.array([1, 1, 0, 0])
    # y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    # print(f1_score(y_true, y_pred))

    pass
