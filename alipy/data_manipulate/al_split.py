"""
Data Split
Split the original dataset into train/test label/unlabelset
Accept not only datamat, but also shape/list of instance name (for image datasets)
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import os

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_array

from ..query_strategy import check_query_type
from ..utils.misc import check_matrix
from ..utils.misc import randperm


def split(X=None, y=None, instance_indexes=None, query_type=None, test_ratio=0.3, initial_label_rate=0.05,
          split_count=10, all_class=True, saving_path='.'):
    """Split given data.

    Provide one of X, y or instance_indexes to execute the split.

    Parameters
    ----------
    X: array-like, optional
        Data matrix with [n_samples, n_features]

    y: array-like, optional
        labels of given data [n_samples, n_labels] or [n_samples]

    instance_indexes: list, optional (default=None)
        List contains instances' names, used for image datasets,
        or provide index list instead of data matrix.
        Must provide one of [instance_names, X, y]

    query_type: str, optional (default='AllLabels')
        Query type. Should be one of:
        'AllLabels': Query all labels of an instance
        'PartLabels': Query part of labels of an instance (Only available in multi-label setting)
        'Features': Query unlab_features of instances

    test_ratio: float, optional (default=0.3)
        Ratio of test set

    initial_label_rate: float, optional (default=0.05)
        Ratio of initial label set
        e.g. Initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        Random split data _split_count times

    all_class: bool, optional (default=True)
        Whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.

    saving_path: str, optional (default='.')
        Giving None to disable saving.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
    """
    # check parameters
    if X is None and y is None and instance_indexes is None:
        raise Exception("Must provide one of X, y or instance_indexes.")
    len_of_parameters = [len(X) if X is not None else None, len(y) if y is not None else None,
                         len(instance_indexes) if instance_indexes is not None else None]
    number_of_instance = np.unique([i for i in len_of_parameters if i is not None])
    if len(number_of_instance) > 1:
        raise ValueError("Different length of instances and _labels found.")
    else:
        number_of_instance = number_of_instance[0]
    if query_type is None:
        query_type = 'AllLabels'
    else:
        if not check_query_type(query_type):
            raise NotImplementedError("Query type %s is not implemented." % type)
    if instance_indexes is not None:
        if not isinstance(instance_indexes, (list, np.ndarray)):
            raise TypeError("A list or np.ndarray object is expected, but received: %s" % str(type(instance_indexes)))
        instance_indexes = np.array(instance_indexes)
    else:
        instance_indexes = np.arange(number_of_instance)

    # split
    train_idx = []
    test_idx = []
    label_idx = []
    unlabel_idx = []
    for i in range(split_count):
        if (not all_class) or y is None:
            rp = randperm(number_of_instance - 1)
            cutpoint = int(round((1 - test_ratio) * len(rp)))
            tp_train = instance_indexes[rp[0:cutpoint]]
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            cutpoint = int(round(initial_label_rate * len(tp_train)))
            if cutpoint <= 1:
                cutpoint = 1
            label_idx.append(tp_train[0:cutpoint])
            unlabel_idx.append(tp_train[cutpoint:])
        else:
            if y is None:
                raise Exception("y must be provided when all_class flag is True.")
            y = check_array(y, ensure_2d=False, dtype=None)
            if y.ndim == 1:
                label_num = len(np.unique(y))
            else:
                label_num = y.shape[1]
            if round((1 - test_ratio) * initial_label_rate * number_of_instance) < label_num:
                raise ValueError(
                    "The initial rate is too small to guarantee that each "
                    "split will contain at least one instance for each class.")

            # check validaty
            while 1:
                rp = randperm(number_of_instance - 1)
                cutpoint = int(round((1 - test_ratio) * len(rp)))
                tp_train = instance_indexes[rp[0:cutpoint]]
                cutpointlabel = int(round(initial_label_rate * len(tp_train)))
                if cutpointlabel <= 1:
                    cutpointlabel = 1
                label_id = tp_train[0:cutpointlabel]
                if y.ndim == 1:
                    if len(np.unique(y[label_id])) == label_num:
                        break
                else:
                    temp = np.sum(y[label_id], axis=0)
                    if not np.any(temp == 0):
                        break
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            label_idx.append(tp_train[0:cutpointlabel])
            unlabel_idx.append(tp_train[cutpointlabel:])

    split_save(train_idx=train_idx, test_idx=test_idx, label_idx=label_idx,
               unlabel_idx=unlabel_idx, path=saving_path)
    return train_idx, test_idx, label_idx, unlabel_idx


def __split_data_matrix(data_matrix=None, matrix_shape=None, test_ratio=0.3, initial_label_rate=0.05,
                        split_count=10, all_class=True, partially_labeled=False, saving_path='.'):
    """Split given data matrix with shape like [n_samples, n_labels or n_features]
    Giving one of matrix or matrix_shape to split the data.

    Parameters
    ----------
    data_matrix: array-like, optional
        Labels of given data, shape like [n_samples, n_labels]

    matrix_shape: tuple, optional (default=None)
        The shape of data_matrix, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of _labels.

    test_ratio: float, optional (default=0.3)
        Ratio of test set

    initial_label_rate: float, optional (default=0.05)
        Ratio of initial label set
        e.g. Initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        Random split data _split_count times

    all_class: bool, optional (default=True)
        Whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.

    partially_labeled: bool, optional (default=False)
        Whether split the data as partially labeled in the multi-label setting.
        If False, the labeled set is fully labeled, otherwise, only part of _labels of each
        instance will be labeled initialized.
        Only available in multi-label setting.

    saving_path: str, optional (default='.')
        Giving None to disable saving.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

    """

    # check parameters
    if data_matrix is None and matrix_shape is None:
        raise Exception("Must provide one of data matrix or matrix_shape.")
    data_shape = None
    if data_matrix is not None:
        data_matrix = check_matrix(data_matrix)
        data_shape = data_matrix.shape
    if matrix_shape is not None:
        if not isinstance(matrix_shape, tuple) and len(matrix_shape) == 2:
            raise TypeError("the shape of data matrix should be a tuple with 2 elements."
                            "The first one is the number of instances, and the other is the"
                            "number of _labels.")
        data_shape = matrix_shape
    instance_indexes = np.arange(data_shape[0])

    # split
    train_idx = []
    test_idx = []
    label_idx = []
    unlabel_idx = []
    for i in range(split_count):
        if partially_labeled:
            # split train test
            rp = randperm(data_shape[0] - 1)
            cutpoint = int(round((1 - test_ratio) * len(rp)))
            tp_train = instance_indexes[rp[0:cutpoint]]

            # split label & unlabel
            train_size = len(tp_train)
            lab_ind = randperm((0, train_size * data_shape[1] - 1), int(round(initial_label_rate * train_size * data_shape[1])))
            if all_class:
                if round(initial_label_rate * train_size) < data_shape[1]:
                    raise ValueError("The initial rate is too small to guarantee that each "
                                     "split will contain at least one instance for each class.")
                while len(np.unique([item % data_shape[1] for item in lab_ind])) != data_shape[1]:
                    # split train test
                    rp = randperm(data_shape[0] - 1)
                    cutpoint = int(round((1 - test_ratio) * len(rp)))
                    tp_train = instance_indexes[rp[0:cutpoint]]
                    # split label & unlabel
                    train_size = len(tp_train)
                    lab_ind = randperm((0, train_size * data_shape[1] - 1), int(round(initial_label_rate * train_size)))
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            unlab_ind = set(np.arange(train_size * data_shape[1]))
            unlab_ind.difference_update(set(lab_ind))
            label_idx.append([(tp_train[item // data_shape[1]], item % data_shape[1]) for item in lab_ind])
            unlabel_idx.append([(tp_train[item // data_shape[1]], item % data_shape[1]) for item in unlab_ind])
        else:
            rp = randperm(data_shape[0] - 1)
            cutpoint = int(round((1 - test_ratio) * len(rp)))
            tp_train = instance_indexes[rp[0:cutpoint]]

            cutpoint_lab = int(round(initial_label_rate * len(tp_train)))
            if cutpoint_lab <= 1:
                cutpoint_lab = 1
            if all_class:
                if cutpoint_lab < data_shape[1]:
                    raise ValueError(
                        "The initial rate is too small to guarantee that each "
                        "split will contain at least one instance-feature pair for each class.")
                while 1:
                    label_id = tp_train[0:cutpoint_lab]
                    temp = np.sum(data_matrix[label_id], axis=0)
                    if not np.any(temp == 0):
                        break
                    rp = randperm(data_shape[0] - 1)
                    cutpoint = int(round((1 - test_ratio) * len(rp)))
                    tp_train = instance_indexes[rp[0:cutpoint]]

                    cutpoint_lab = int(round(initial_label_rate * len(tp_train)))
            train_idx.append(tp_train)
            test_idx.append(instance_indexes[rp[cutpoint:]])
            label_idx.append([(i,) for i in tp_train[0:cutpoint_lab]])
            unlabel_idx.append([(i,) for i in tp_train[cutpoint_lab:]])
    split_save(train_idx=train_idx, test_idx=test_idx, label_idx=label_idx,
               unlabel_idx=unlabel_idx, path=saving_path)
    return train_idx, test_idx, label_idx, unlabel_idx


def split_features(feature_matrix=None, feature_matrix_shape=None, test_ratio=0.3, missing_rate=0.5,
                   split_count=10, all_features=True, saving_path='.'):
    """
    Split given feature matrix in feature querying setting.
    Giving one of feature_matrix or feature_matrix_shape to split the data.

    The matrix will be split randomly in _split_count times, the testing set
    is the set of instances with complete feature vectors. The training set
    has missing feature with the rate of missing_rate.

    Parameters
    ----------
    feature_matrix: array-like, optional
        Feature matrix, shape [n_samples, n_labels].

    feature_matrix_shape: tuple, optional (default=None)
        The shape of data_matrix, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of feature.

    test_ratio: float, optional (default=0.3)
        Ratio of test set.

    missing_rate: float, optional (default=0.5)
        Ratio of missing value.

    split_count: int, optional (default=10)
        Random split data split_count times

    all_features: bool, optional (default=True)
        Whether each split will contain at least one instance for each feature.
        If False, a totally random split will be performed.

    saving_path: str, optional (default='.')
        Giving None to disable saving.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

    """
    return __split_data_matrix(data_matrix=feature_matrix, matrix_shape=feature_matrix_shape, test_ratio=test_ratio,
                               initial_label_rate=1-missing_rate, split_count=split_count,
                               all_class=all_features, partially_labeled=True, saving_path=saving_path)


def split_multi_label(y=None, label_shape=None, test_ratio=0.3, initial_label_rate=0.05,
                      split_count=10, all_class=True, saving_path='.'):
    """Split given data matrix with shape like [n_samples, n_labels]
    Giving one of matrix or matrix_shape to split the data.

    Parameters
    ----------
    y: array-like, optional
        Labels of given data, shape like [n_samples, n_labels]

    label_shape: tuple, optional (default=None)
        The shape of data_matrix, should be a tuple with 2 elements.
        The first one is the number of instances, and the other is the
        number of _labels.

    test_ratio: float, optional (default=0.3)
        Ratio of test set

    initial_label_rate: float, optional (default=0.05)
        Ratio of initial label set
        e.g. Initial_labelset*(1-test_ratio)*n_samples

    split_count: int, optional (default=10)
        Random split data _split_count times

    all_class: bool, optional (default=True)
        Whether each split will contain at least one instance for each class.
        If False, a totally random split will be performed.

    saving_path: str, optional (default='.')
        Giving None to disable saving.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

    """
    return __split_data_matrix(data_matrix=y, matrix_shape=label_shape, test_ratio=test_ratio,
                        initial_label_rate=initial_label_rate,
                        split_count=split_count, all_class=all_class, partially_labeled=False,
                        saving_path=saving_path)


def split_load(path):
    """Load split from path.

    Parameters
    ----------
    path: str
        Path to a dir which contains train_idx.txt, test_idx.txt, label_idx.txt, unlabel_idx.txt.

    Returns
    -------
    train_idx: list
        index of training set, shape like [n_split_count, n_training_samples]

    test_idx: list
        index of testing set, shape like [n_split_count, n_testing_samples]

    label_idx: list
        index of labeling set, shape like [n_split_count, n_labeling_samples]

    unlabel_idx: list
        index of unlabeling set, shape like [n_split_count, n_unlabeling_samples]
    """
    if not isinstance(path, str):
        raise TypeError("A string is expected, but received: %s" % str(type(path)))
    saving_path = os.path.abspath(path)
    if not os.path.isdir(saving_path):
        raise Exception("A path to a directory is expected.")

    ret_arr = []
    for fname in ['train_idx.txt', 'test_idx.txt', 'label_idx.txt', 'unlabel_idx.txt']:
        if not os.path.exists(os.path.join(saving_path, fname)):
            if os.path.exists(os.path.join(saving_path, fname.split()[0] + '.npy')):
                ret_arr.append(np.load(os.path.join(saving_path, fname.split()[0] + '.npy')))
            else:
                ret_arr.append(None)
        else:
            ret_arr.append(np.loadtxt(os.path.join(saving_path, fname)))
    return ret_arr[0], ret_arr[1], ret_arr[2], ret_arr[3]


def split_save(train_idx, test_idx, label_idx, unlabel_idx, path):
    """Save the split to file for auditting or loading for other methods.

    Parameters
    ----------
    saving_path: str
        path to save the settings. If a dir is not provided, it will generate a folder called
        'alipy_split' for saving.

    """
    if path is None:
        return
    else:
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))

    saving_path = os.path.abspath(path)
    if os.path.isdir(saving_path):
        np.savetxt(os.path.join(saving_path, 'train_idx.txt'), train_idx, fmt='%d')
        np.savetxt(os.path.join(saving_path, 'test_idx.txt'), test_idx, fmt='%d')
        if len(np.shape(label_idx)) == 2:
            np.savetxt(os.path.join(saving_path, 'label_idx.txt'), label_idx, fmt='%d')
            np.savetxt(os.path.join(saving_path, 'unlabel_idx.txt'), unlabel_idx, fmt='%d')
        else:
            np.save(os.path.join(saving_path, 'label_idx.npy'), label_idx)
            np.save(os.path.join(saving_path, 'unlabel_idx.npy'), unlabel_idx)
    else:
        raise Exception("A path to a directory is expected.")
