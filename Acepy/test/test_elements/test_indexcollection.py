"""
Test the functions in index container:
1. IndexCollection
2. MultiLabelIndexCollection (FeatureIndexCollection)
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause


from __future__ import division
import pytest
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels, type_of_target
from utils.al_collections import *
from utils.ace_warnings import *
from data_process.al_split import split_multi_label

index1 = IndexCollection([1, 2, 3])
index2 = IndexCollection([1, 2, 2, 3])
multi_lab_ind1 = MultiLabelIndexCollection([(0, 1), (0, 2), (0, (3, 4)), (1, (0, 1))], label_size=5)


def test_basic_ind1():
    for item in index1:
        assert item in index2
    for item in index2:
        assert item in index1
    assert 1 in index2
    assert len(index1) == 3
    index1.add(4)
    assert (len(index1) == 4)
    index1.discard(4)
    assert index1.index == [1, 2, 3]
    index1.update([4, 5])
    assert index1.index == [1, 2, 3, 4, 5]
    index1.difference_update([4, 5])
    assert index1.index == [1, 2, 3]
    assert len(index1.random_sampling(0.66)) == 2


def test_warn_ind1():
    with pytest.warns(RepeatElementWarning, match=r'.*same elements in the given data'):
        a = IndexCollection([1, 2, 2, 3])
    with pytest.warns(RepeatElementWarning, match=r'.*has already in the collection.*'):
        a.add(3)
    a.add(4)
    with pytest.warns(InexistentElementWarning, match=r'.*to discard is not in the collection.*'):
        a.discard(6)
    assert a.pop() == 4
    with pytest.warns(RepeatElementWarning, match=r'.*has already in the collection.*'):
        a.update(IndexCollection([2, 9, 10]))
    with pytest.warns(InexistentElementWarning, match=r'.*to discard is not in the collection.*'):
        a.difference_update(IndexCollection([2, 100]))


def test_raise_ind1():
    # with pytest.raises(TypeError, match='Different types found in the given _indexes.'):
    #     a = IndexCollection([1, 0.5, ])
    b = IndexCollection([1, 2, 3, 4])
    with pytest.raises(TypeError, match=r'.*parameter is expected, but received.*'):
        b.update([0.2, 0.5])


def test_basic_multiind():
    assert len(multi_lab_ind1) == 6
    multi_lab_ind1.update((0, 0))
    assert len(multi_lab_ind1) == 7
    assert (0, 0) in multi_lab_ind1
    multi_lab_ind1.discard((0, 0))
    assert len(multi_lab_ind1) == 6
    assert (0, 0) not in multi_lab_ind1
    multi_lab_ind1.update([(1, 2), (1, (3, 4))])
    assert (1, 3) in multi_lab_ind1
    multi_lab_ind1.update([(2,)])
    assert (2, 0) in multi_lab_ind1
    with pytest.warns(InexistentElementWarning):
        multi_lab_ind1.difference_update([(0,)])
    assert (0, 1) not in multi_lab_ind1


def test_warn_multiind():
    with pytest.warns(RepeatElementWarning, match=r'.*same elements in the given data'):
        a = MultiLabelIndexCollection([(0,1), (0,2), (0,1)], label_size=3)
    with pytest.warns(ValidityWarning, match=r'This collection does not have a label_size value.*'):
        MultiLabelIndexCollection()
    with pytest.warns(RepeatElementWarning):
        a.update((0, 1))
    with pytest.warns(InexistentElementWarning):
        a.discard((0,0))


def test_raise_multiind():
    with pytest.raises(ValueError, match=r'.*out of bound.*'):
        multi_lab_ind1.add((0, 100))
    with pytest.raises(AssertionError):
        multi_lab_ind1.add(dict({1:1}))
    with pytest.raises(TypeError):
        multi_lab_ind1.update(dict({1: 1}))

