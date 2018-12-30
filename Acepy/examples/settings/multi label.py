import copy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from acepy.query_strategy.multi_label import *
from acepy.index.multi_label_tools import get_Xy_in_multilabel
from acepy import ToolBox

X, y = load_iris(return_X_y=True)
mlb = OneHotEncoder()
mult_y = mlb.fit_transform(y.reshape((-1,1)))
mult_y = np.asarray(mult_y.todense())
mult_y[mult_y == 0] = -1

acebox = ToolBox(X=X, y=mult_y, query_type='PartLabels')
acebox.split_AL(test_ratio=0.2, initial_label_rate=0.05, all_class=False)

AUDI_result = []
QUIRE_result = []
MMC_result = []
Adaptive_result = []
Random_result = []


def main_loop(acebox, round, strategy):
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)
    query_y = mult_y.copy()
    # base model
    model = LabelRankingModel()

    while len(label_ind) <= 120:
        # query and update
        select_labs = strategy.select(label_ind, unlab_ind)
        label_ind.update(select_labs)
        unlab_ind.difference_update(select_labs)

        # train/test
        X_tr, y_tr, _ = get_Xy_in_multilabel(label_ind, X=X, y=y)
        model.fit(X=X_tr, y=y_tr)
        pres, pred = model.predict(X[test_idx])

        perf = acebox.calc_performance_metric(y_true=mult_y[test_idx], y_pred=pred, performance_metric='hamming_loss')

        # save
        st = acebox.State(select_index=select_labs, performance=perf)
        saver.add_state(st)

    return copy.deepcopy(saver)


for round in range(5):
    # init strategies
    audi = QueryMultiLabelAUDI(X, y)
    quire = QueryMultiLabelQUIRE(X, y)
    mmc = QueryMultiLabelMMC(X, y)
    adaptive = QueryMultiLabelAdaptive(X, y)
    random = QueryMultiLabelRandom()




