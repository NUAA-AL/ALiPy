import sys
sys.path.append(r'D:\al_tools\ALiPy\ALiPy')
import copy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from alipy.query_strategy.multi_label import *
from alipy.index.multi_label_tools import get_Xy_in_multilabel
from alipy import ToolBox

X, y = load_iris(return_X_y=True)
mlb = OneHotEncoder()
mult_y = mlb.fit_transform(y.reshape((-1,1)))
mult_y = np.asarray(mult_y.todense())
mult_y[mult_y == 0] = -1

alibox = ToolBox(X=X, y=mult_y, query_type='PartLabels')
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.05, all_class=False)

def main_loop(alibox, round, strategy):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    query_y = mult_y.copy()
    # base model
    model = LabelRankingModel()

    while len(label_ind) <= 120:
        # query and update
        select_labs = strategy.select(label_ind, unlab_ind)
        # use cost to record the amount of queried instance-label pairs
        if len(select_labs[0]) == 1:
            cost = mult_y.shape[1]
        else:
            cost = len(select_labs)
        label_ind.update(select_labs)
        unlab_ind.difference_update(select_labs)

        # train/test
        X_tr, y_tr, _ = get_Xy_in_multilabel(label_ind, X=X, y=mult_y)
        model.fit(X=X_tr, y=y_tr)
        pres, pred = model.predict(X[test_idx])
        perf = alibox.calc_performance_metric(y_true=mult_y[test_idx], y_pred=pred, performance_metric='hamming_loss')

        # save
        st = alibox.State(select_index=select_labs, performance=perf, cost=cost)
        saver.add_state(st)

    return copy.deepcopy(saver)

audi_result = []
quire_result = []
random_result = []
mmc_result = []
adaptive_result = []

for round in range(5):
    # init strategies
    audi = QueryMultiLabelAUDI(X, mult_y)
    quire = QueryMultiLabelQUIRE(X, mult_y)
    mmc = QueryMultiLabelMMC(X, mult_y)
    adaptive = QueryMultiLabelAdaptive(X, mult_y)
    random = QueryMultiLabelRandom()

    audi_result.append(main_loop(alibox, round, strategy=audi))
    quire_result.append(main_loop(alibox, round, strategy=quire))
    mmc_result.append(main_loop(alibox, round, strategy=mmc))
    adaptive_result.append(main_loop(alibox, round, strategy=adaptive))
    random_result.append(main_loop(alibox, round, strategy=random))

analyser = alibox.get_experiment_analyser(x_axis='cost')
analyser.add_method(method_name='AUDI', method_results=audi_result)
analyser.add_method(method_name='QUIRE', method_results=quire_result)
analyser.add_method(method_name='RANDOM', method_results=random_result)
analyser.add_method(method_name='MMC', method_results=mmc_result)
analyser.add_method(method_name='Adaptive', method_results=adaptive_result)
analyser.plot_learning_curves()
