"""
Test the functions in StateIO class
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import os

import numpy as np
import pytest
from sklearn.datasets import load_iris

from alipy.experiment import State, StateIO
from alipy.toolbox import ToolBox as acebox

X, y = load_iris(return_X_y=True)
split_count = 5
cur_path = os.path.abspath('.')
toolbox = acebox(X=X, y=y, query_type='AllLabels', saving_path=cur_path)

# split data
toolbox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)
saver = toolbox.get_stateio(round=0)
saver.init_L.difference_update([0, 1, 2])
saver.init_U.update([0, 1, 2])

st1_batch2 = State(select_index=[0, 1], performance=0.89)
st1_batch1 = State(select_index=[1], performance=0.89)
st2_batch1 = State(select_index=[0], performance=0.89)
st3_batch1 = State(select_index=[2], performance=0.89)


def test_stateio_validity_checking():
    saver.add_state(st1_batch1)
    saver.add_state(st1_batch2)
    saver.add_state(st2_batch1)
    assert not saver.check_batch_size()
    assert saver.cost_inall == 0
    nq, cost = saver.refresh_info()
    assert nq == 4
    assert cost == 0


def test_stateio_basic():
    saver.set_initial_point(0.88)
    st = saver.get_state(0)
    assert st.get_value('select_index') == st1_batch1.get_value('select_index')
    assert st.get_value('performance') == st1_batch1.get_value('performance')
    st = saver.pop(1)
    assert st.get_value('select_index') == st1_batch2.get_value('select_index')
    assert st.get_value('performance') == st1_batch2.get_value('performance')
    assert len(saver) == 2
    saver.add_state(st3_batch1)


def test_stateio_recover():
    train_ini, test_ini, L_ini, U_ini = saver.get_workspace(0)
    assert 0 not in L_ini
    assert 1 not in L_ini
    assert 2 not in L_ini
    train, test, L, U = saver.get_workspace(2)
    assert 0 in L
    assert 1 in L
    assert 2 not in L
    assert len(L) - len(L_ini) == 2
    assert (len(saver) == 3)
    train, test, L, U = saver.recover_workspace(2)
    assert 0 in L
    assert 1 in L
    assert 2 not in L
    assert len(L) - len(L_ini) == 2
    assert(len(saver)==2)
    with pytest.raises(AssertionError):
        saver.get_workspace(5)


def test_stateio_output_file():
    saver.save()
    assert os.path.exists(os.path.join(cur_path,'AL_round_0.pkl'))


def test_stateio_input_file():
    saver2 = StateIO.load(os.path.join(cur_path,'AL_round_0.pkl'))
    assert len(saver2) == len(saver)
    tr,te,L,U = saver2.get_workspace()
    tr2, te2, L2, U2 =saver.get_workspace()
    assert np.all(L.index==L2.index)
    assert np.all(U.index==U2.index)
    assert np.all (tr==tr2)
    assert np.all(te==te2)

