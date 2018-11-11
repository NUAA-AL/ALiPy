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
    # with pytest.raises(TypeError):
    #     check_index_multilabel((1, 2, 0.5, True))


def test_check_one_to_one_correspondence():
    assert check_one_to_one_correspondence([i for i in range(10)], [i for i in range(10)], [i for i in range(10)])
    assert not check_one_to_one_correspondence([i for i in range(10)], [i for i in range(9)], [i for i in range(10)])
    assert check_one_to_one_correspondence([i for i in range(10)], [i for i in range(10)], [i for i in range(10)])
    a = np.array([i for i in range(50)])
    assert not check_one_to_one_correspondence(np.reshape(a, (5, 10)), np.reshape(a, (10, 5)))


def test_nlargestarg_nsmallestarg():
    a = np.array([1, 2, 3, 4, 5])
    assert a[nlargestarg(a, 1)] == np.array([5])
    assert a[nsmallestarg(a, 1)] == np.array([1])
    assert set(nlargestarg(a, 2)) == {4,3}
    assert set(nlargestarg(a, 3)) == {2,4,3}
    assert set(nsmallestarg(a, 2)) == {1,0}
    assert set(nsmallestarg(a, 3)) == {1,2, 0}


    
