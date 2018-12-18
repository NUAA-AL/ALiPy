import scipy.io as scio

from acepy import ToolBox
from acepy.query_strategy.multi_label import QueryMultiLabelAUDI
from acepy.query_strategy.query_type import QueryTypeAURO

ld = scio.loadmat('C:\\git\\AUDI\\generate_data.mat')
train_data = ld['train_data']
train_targets = ld['train_targets']
train_targets = train_targets[:, 1:-1]

acebox = ToolBox(X=train_data, y=train_targets, query_type='PartLabels')
acebox.split_AL(test_ratio=0.2, initial_label_rate=0.1, all_class=False)

for round in range(10):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    audi = QueryMultiLabelAUDI(X=train_data, y=train_targets, initial_labeled_indexes=label_ind)
    auro = QueryTypeAURO(X=train_data, y=train_targets, initial_labeled_indexes=label_ind)
    while True:
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind, select_ora = audi.select(label_ind, unlab_ind)
        print(select_ind)
        print(select_ora)
        select_ind, y1, y2 = auro.select(label_ind, unlab_ind)
        print(select_ind)
        print(y1, y2)
        pass
