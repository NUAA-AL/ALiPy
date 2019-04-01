import copy

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, normalize

from alipy import ToolBox
from alipy.index.multi_label_tools import get_Xy_in_multilabel
from alipy.query_strategy.multi_label import *

X, y = load_iris(return_X_y=True)
X = normalize(X, norm='l2')
mlb = OneHotEncoder()
mult_y = mlb.fit_transform(y.reshape((-1, 1)))
mult_y = np.asarray(mult_y.todense())
mult_y_for_metric = mult_y.copy()

# Or generate a dataset with any sizes
# X, mult_y = make_multilabel_classification(n_samples=5000, n_features=20, n_classes=5, length=5)

# Since we are using the label ranking model, the label 0 means unknown. we need to
# set the 0 entries to -1 which means irrelevant.
mult_y[mult_y == 0] = -1

alibox = ToolBox(X=X, y=mult_y, query_type='PartLabels')
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.05, all_class=False)


def main_loop(alibox, round, strategy):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    # base model
    model = LabelRankingModel()

    ini_lab_num = len(label_ind)
    # A simple stopping criterion to specify the query budget.
    while len(label_ind) - ini_lab_num <= 120:
        # query and update
        if isinstance(strategy, QueryMultiLabelAUDI):
            # If you are using a label ranking model, pass it to AUDI. It can
            # avoid re-training a label ranking model inside the algorithm
            select_labs = strategy.select(label_ind, unlab_ind, model=model)
        else:
            select_labs = strategy.select(label_ind, unlab_ind)
        # use cost to record the amount of queried instance-label pairs
        if len(select_labs[0]) == 1:
            cost = mult_y.shape[1]
        else:
            cost = len(select_labs)
        label_ind.update(select_labs)
        unlab_ind.difference_update(select_labs)

        # train/test
        X_tr, y_tr, _ = get_Xy_in_multilabel(select_labs, X=X, y=mult_y, unknown_element=0)
        model.fit(X=X_tr, y=y_tr, is_incremental=True)
        pres, pred = model.predict(X[test_idx])
        # using sklearn to calc micro-f1
        pred[pred == -1] = 0
        perf = f1_score(y_true=mult_y_for_metric[test_idx], y_pred=pred, average='micro')

        # save
        st = alibox.State(select_index=select_labs, performance=perf, cost=cost)
        saver.add_state(st)
        saver.save()

    return copy.deepcopy(saver)


audi_result = []
quire_result = []
random_result = []
mmc_result = []
adaptive_result = []

for round in range(3):
    # init strategies
    audi = QueryMultiLabelAUDI(X, mult_y)
    quire = QueryMultiLabelQUIRE(X, mult_y, kernel='rbf')
    mmc = QueryMultiLabelMMC(X, mult_y)
    adaptive = QueryMultiLabelAdaptive(X, mult_y)
    random = QueryMultiLabelRandom(select_type='ins')

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
analyser.plot_learning_curves(plot_interval=20)  # plot a performance point in every 20 queries of instance-label pairs
