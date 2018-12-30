import os
from sklearn.datasets import load_iris
from acepy.experiment import State, StateIO
from acepy.toolbox import ToolBox

X, y = load_iris(return_X_y=True)
split_count = 5
cur_path = os.path.abspath('.')
toolbox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=cur_path)

# split data
toolbox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)
train_ind, test_ind, L_ind, U_ind = toolbox.get_split(round=0)
# -------Initialize StateIO----------
saver = StateIO(round=0, train_idx=train_ind, test_idx=test_ind, init_L=L_ind, init_U=U_ind, saving_path='.')
# or by using toolbox 
# saver = toolbox.get_stateio(round=0)

saver.init_L.difference_update([0, 1, 2])
saver.init_U.update([0, 1, 2])

# -------Basic operations------------
st1_batch1 = State(select_index=[1], performance=0.89)
my_value = 'my_entry_info'
st1_batch1.add_element(key='my_entry', value=my_value)
st1_batch2 = State(select_index=[0, 1], performance=0.89)
st2_batch1 = State(select_index=[0], performance=0.89)
st3_batch1 = State(select_index=[2], performance=0.89)

saver.add_state(st1_batch1)
saver.add_state(st1_batch2)
saver.add_state(st2_batch1)

saver.save()

prev_st = saver.get_state(index=1) # get 2nd query
# or use the index operation directly
prev_st = saver[1]

value = prev_st.get_value(key='select_index')
# or use the index operation directly
value = prev_st['select_index']

# ---------Recover workspace---------
train, test, L, U = saver.get_workspace(iteration=1)
# or recover the saver itself
saver.recover_workspace(iteration=1)

saver = StateIO.load(path='./AL_round_0.pkl')
train, test, L, U = saver.get_workspace() # will return the latest workspace
