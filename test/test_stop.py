"""
Test the functions in StateIO class
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause

from __future__ import division

import copy

from alipy.experiment import StoppingCriteria, StateIO, State

stop1 = StoppingCriteria()
stop2 = StoppingCriteria(stopping_criteria='num_of_queries', value=10)
stop3 = StoppingCriteria(stopping_criteria='cost_limit', value=10)
stop4 = StoppingCriteria(stopping_criteria='percent_of_unlabel', value=0.1)
stop5 = StoppingCriteria(stopping_criteria='time_limit', value=2)
example_saver = StateIO(round=0, train_idx=list(range(8)), test_idx=[8,9,10], init_L=[0,1], init_U=[2,3,4,5,6,7])


# def test_stop5():
#     assert not stop5.is_stop()
#     stop5._start_time -= 1
#     assert not stop5.is_stop()
#     stop5._start_time -= 1
#     assert stop5.is_stop()

def test_stop1():
    assert not stop1.is_stop()
    stop1.update_information(example_saver)
    assert not stop1.is_stop()
    # example_saver_local = copy.deepcopy(example_saver)
    # example_saver_local.add_state(State(select_index=[2,3,4,5,6,7], performance=0.89))
    stop1.update_information(StateIO(round=0, train_idx=list(range(8)), test_idx=[8,9,10], init_L=list(range(8)), init_U=[]))
    assert stop1.is_stop()


def test_stop2():
    assert not stop2.is_stop()
    stop2.update_information(example_saver)
    example_saver_local = copy.deepcopy(example_saver)
    assert stop2._current_iter == 0
    example_saver_local.add_state(State(select_index=[2], performance=0.89))
    stop2.update_information(example_saver_local)
    assert stop2._current_iter == 1
    assert not stop2.is_stop()
    stop2._current_iter = 10
    assert stop2.is_stop()


def test_stop3():
    assert not stop3.is_stop()
    stop3.update_information(example_saver)
    example_saver_local = copy.deepcopy(example_saver)
    assert stop3._accum_cost == 0
    example_saver_local.add_state(State(select_index=[2], performance=0.89, cost=[3]))
    stop3.update_information(example_saver_local)
    assert stop3._accum_cost == 3
    assert not stop3.is_stop()
    example_saver_local.add_state(State(select_index=[3], performance=0.89, cost=[7]))
    stop3.update_information(example_saver_local)
    assert stop3._accum_cost == 10
    assert stop3.is_stop()


def test_stop4():
    assert not stop4.is_stop()
    stop4.update_information(example_saver)
    example_saver_local = copy.deepcopy(example_saver)
    assert stop4._percent == 0
    example_saver_local.add_state(State(select_index=[2], performance=0.89, cost=[3]))
    stop4.update_information(example_saver_local)
    assert stop4._percent == 1/6
    assert stop4.is_stop()

