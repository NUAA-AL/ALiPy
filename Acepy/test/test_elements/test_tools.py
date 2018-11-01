from __future__ import division

import numpy as np
import pytest

from acepy.index.multi_label_tools import check_index_multilabel
from acepy.utils.misc import check_one_to_one_correspondence, nlargestarg, nsmallestarg


def test_check_index_multilabel():

    assert [(1, 2, 3, 4, 5)] == check_index_multilabel((1, 2, 3, 4, 5))
    print(check_index_multilabel((1, 2, 0.5, True)))
    with pytest.raises(TypeError):
        check_index_multilabel([0])
    with pytest.raises(TypeError):
        check_index_multilabel((1, 2, 0.5, True))


def test_check_one_to_one_correspondence():
    assert check_one_to_one_correspondence([i for i in range(10)], [i for i in range(10)], [i for i in range(10)])
    assert not check_one_to_one_correspondence([i for i in range(10)], [i for i in range(9)], [i for i in range(10)])
    assert check_one_to_one_correspondence([i for i in range(10)], [i for i in range(10)], [i for i in range(10)])
    a = np.array([i for i in range(50)])
    assert not check_one_to_one_correspondence(np.reshape(a, (5, 10)), np.reshape(a, (10, 5)))


def test_nlargestarg_nsmallestarg():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 1, 1, 1, 1])
    print(a[nlargestarg(a, 1)])
    assert a[nlargestarg(a, 1)] == np.array([5])
    print(a[nlargestarg(a, 2)])
    assert a[nlargestarg(a, 2)] == np.array([5, 4])
    assert a[nlargestarg(b, 3)] == np.array([1, 1, 1])
    assert a[nsmallestarg(a, 1)] == np.array([1])
    assert a[nsmallestarg(a, 2)] == np.array([1, 2])
    assert a[nsmallestarg(b, 3)] == np.array([1, 1, 1])

    
