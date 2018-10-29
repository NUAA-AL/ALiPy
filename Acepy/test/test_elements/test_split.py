"""
Test the functions in al split modules
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels, type_of_target
from data_process.al_split import *
from utils.tools import check_index_multilabel, integrate_multilabel_index, flattern_multilabel_index

"""
Test 3 types of split setting:
1. common setting with int indexes.
2. multi-label setting with the fully labeled warm start (index is tuple)
3. split feature matrix to discard some values randomly (similar to multi-label).
"""

X, y = load_iris(return_X_y=True)
mult_y = LabelBinarizer().fit_transform(y=y)
split_count = 10
instance_num = 150
feature_num = 4
label_num = 3


def test_split1():
    train_idx, test_idx, label_idx, unlabel_idx = split(X=X,
                                                        y=y,
                                                        all_class=False, split_count=split_count,
                                                        test_ratio=0.3, initial_label_rate=0.05,
                                                        saving_path=None,
                                                        query_type='AllLabels')
    assert len(train_idx) == split_count
    assert len(test_idx) == split_count
    assert len(label_idx) == split_count
    assert len(unlabel_idx) == split_count

    for i in range(split_count):
        train = set(train_idx[i])
        test = set(test_idx[i])
        lab = set(label_idx[i])
        unl = set(unlabel_idx[i])

        assert len(test) == round(0.3 * instance_num)
        assert len(lab) == round(0.05 * len(train))

        # validity
        traintest = train.union(test)
        labun = lab.union(unl)
        assert traintest == set(range(instance_num))
        assert labun == train


def test_split1_allclass():
    train_idx, test_idx, label_idx, unlabel_idx = split(X=X,
                                                        y=y,
                                                        all_class=True, split_count=split_count,
                                                        test_ratio=0.3, initial_label_rate=0.05,
                                                        saving_path=None,
                                                        query_type='AllLabels')
    assert len(train_idx) == split_count
    assert len(test_idx) == split_count
    assert len(label_idx) == split_count
    assert len(unlabel_idx) == split_count

    for i in range(split_count):
        train = set(train_idx[i])
        test = set(test_idx[i])
        lab = set(label_idx[i])
        unl = set(unlabel_idx[i])

        assert len(test) == round(0.3 * instance_num)
        assert len(lab) == round(0.05 * len(train))

        # validity
        traintest = train.union(test)
        labun = lab.union(unl)
        assert traintest == set(range(instance_num))
        assert labun == train

        # is all-class
        len(unique_labels(y[label_idx[i]])) == label_num


def test_split1_allclass_assert():
    with pytest.raises(ValueError, match=r'The initial rate is too small to guarantee that each.*'):
        train_idx, test_idx, label_idx, unlabel_idx = split(X=X,
                                                            y=y,
                                                            all_class=True, split_count=split_count,
                                                            test_ratio=0.3, initial_label_rate=0.01,
                                                            saving_path=None,
                                                            query_type='AllLabels')


def test_split2():
    with pytest.raises(TypeError):
        train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
            y=y, label_shape=(instance_num, label_num),
            all_class=False, split_count=split_count,
            test_ratio=0.3, initial_label_rate=0.05,
            saving_path=None
        )
    train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
        y=mult_y, label_shape=(instance_num, label_num),
        all_class=False, split_count=split_count,
        test_ratio=0.3, initial_label_rate=0.05,
        saving_path=None
    )
    assert len(train_idx) == split_count
    assert len(test_idx) == split_count
    assert len(label_idx) == split_count
    assert len(unlabel_idx) == split_count
    for i in range(split_count):
        check_index_multilabel(label_idx[i])
        check_index_multilabel(unlabel_idx[i])
        train = set(train_idx[i])
        test = set(test_idx[i])
        assert len(test) == round(0.3 * instance_num)

        len(label_idx[i]) == len(integrate_multilabel_index(label_idx[i], label_size=label_num))
        # validity
        lab = set([j[0] for j in label_idx[i]])
        assert len(lab) == round(0.05 * len(train))

        unl = set([j[0] for j in unlabel_idx[i]])
        traintest = train.union(test)
        labun = lab.union(unl)

        assert traintest == set(range(instance_num))
        assert labun == train


def test_split2_allclass():
    train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
        y=mult_y, label_shape=(instance_num, label_num),
        all_class=True, split_count=split_count,
        test_ratio=0.3, initial_label_rate=0.05,
        saving_path=None
    )
    assert len(train_idx) == split_count
    assert len(test_idx) == split_count
    assert len(label_idx) == split_count
    assert len(unlabel_idx) == split_count
    for i in range(split_count):
        check_index_multilabel(label_idx[i])
        check_index_multilabel(unlabel_idx[i])
        train = set(train_idx[i])
        test = set(test_idx[i])

        assert len(label_idx[i]) == len(integrate_multilabel_index(label_idx[i], label_size=label_num))
        # validity
        lab = set([j[0] for j in label_idx[i]])
        unl = set([j[0] for j in unlabel_idx[i]])
        traintest = train.union(test)
        labun = lab.union(unl)

        assert len(test) == round(0.3 * instance_num)
        assert len(lab) == round(0.05 * len(train))
        assert traintest == set(range(instance_num))
        assert labun == train


def test_split3():
    train_idx, test_idx, label_idx, unlabel_idx = split_features(feature_matrix=X, feature_matrix_shape=X.shape,
                                                                 test_ratio=0.3, missing_rate=0.2,
                                                                 split_count=split_count,
                                                                 all_features=False,
                                                                 saving_path=None)
    assert len(train_idx) == split_count
    assert len(test_idx) == split_count
    assert len(label_idx) == split_count
    assert len(unlabel_idx) == split_count
    for i in range(split_count):
        train = set(train_idx[i])
        test = set(test_idx[i])
        traintest = train.union(test)

        # validity
        assert len(flattern_multilabel_index(index_arr=unlabel_idx[i], label_size=feature_num)) == round(
            0.2 * len(train) * feature_num)
        assert len(test) == round(0.3 * instance_num)

        assert traintest == set(range(instance_num))
        assert len(
            [j[0] for j in integrate_multilabel_index(label_idx[i] + unlabel_idx[i], label_size=feature_num)]) == len(
            train_idx[i])

def test_split3_all_features():
    train_idx, test_idx, label_idx, unlabel_idx = split_features(feature_matrix=X, feature_matrix_shape=X.shape,
                                                                 test_ratio=0.3, missing_rate=0.2,
                                                                 split_count=split_count,
                                                                 all_features=True,
                                                                 saving_path=None)
    assert len(train_idx) == split_count
    assert len(test_idx) == split_count
    assert len(label_idx) == split_count
    assert len(unlabel_idx) == split_count
    for i in range(split_count):
        train = set(train_idx[i])
        test = set(test_idx[i])
        traintest = train.union(test)

        # validity
        assert len(flattern_multilabel_index(index_arr=unlabel_idx[i], label_size=feature_num)) == round(
            0.2 * len(train) * X.shape[1])
        assert len(test) == round(0.3 * instance_num)

        assert traintest == set(range(instance_num))
        assert len(
            [j[0] for j in integrate_multilabel_index(label_idx[i] + unlabel_idx[i], label_size=feature_num)]) == len(
            train_idx[i])
