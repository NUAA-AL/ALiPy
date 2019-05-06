import copy
import numpy as np
from sklearn.datasets import load_iris, make_multilabel_classification
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.metrics import f1_score
from alipy.query_strategy.query_type import QueryTypeAURO
from alipy.query_strategy.multi_label import LabelRankingModel
from alipy.index.multi_label_tools import get_Xy_in_multilabel
from alipy import ToolBox

X, y = load_iris(return_X_y=True)
# X, mult_y = make_multilabel_classification(n_samples=2500, n_labels=3, n_classes=10, n_features=15)
mlb = OneHotEncoder()
mult_y = mlb.fit_transform(y.reshape((-1, 1)))
mult_y = np.asarray(mult_y.todense())
X = normalize(X, norm='l2')
mult_y_for_metric = mult_y.copy()
mult_y[mult_y == 0] = -1

alibox = ToolBox(X=X, y=mult_y, query_type='PartLabels')
alibox.split_AL(test_ratio=0.2, initial_label_rate=0.05, all_class=False)
model = LabelRankingModel() # base model

# query type strategy
AURO_results = []

for round in range(5):

    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    query_y = mult_y.copy() # for labeling `less relevant`
    AURO_strategy = QueryTypeAURO(X=X, y=mult_y)
    # init model
    X_tr, y_tr, _ = get_Xy_in_multilabel(label_ind, X=X, y=mult_y)
    model.fit(X=X_tr, y=y_tr)

    for iter in range(100):

        select_ins, select_y1, select_y2 = AURO_strategy.select(label_ind, unlab_ind, model=model, y_mat=query_y)

        # relevance
        y1 = mult_y[select_ins, select_y1]
        y2 = mult_y[select_ins, select_y2]
        if y1 < 0 and y2 < 0:
            query_y[select_ins, select_y1] = -1
            query_y[select_ins, select_y2] = -1
        elif y1 > y2:
            query_y[select_ins, select_y1] = 1
            query_y[select_ins, select_y2] = 0.5
        else:
            query_y[select_ins, select_y1] = 0.5
            query_y[select_ins, select_y2] = 1

        # record results
        label_ind.update([(select_ins, select_y1), (select_ins, select_y2)])
        unlab_ind.difference_update([(select_ins, select_y1), (select_ins, select_y2)])

        # train/test
        X_tr, y_tr, _ = get_Xy_in_multilabel([(select_ins, select_y1), (select_ins, select_y2)], X=X, y=query_y, unknown_element=0)
        model.fit(X=X_tr, y=y_tr, is_incremental=True)
        pres, pred = model.predict(X[test_idx])

        # using sklearn to calc micro-f1
        pred[pred == -1] = 0
        perf = f1_score(y_true=mult_y_for_metric[test_idx], y_pred=pred, average='micro')

        # save
        st = alibox.State(select_index=[(select_ins, select_y1), (select_ins, select_y2)], performance=perf)
        saver.add_state(st)


    AURO_results.append(copy.copy(saver))

analyser = alibox.get_experiment_analyser()
analyser.add_method(method_name='AURO', method_results=AURO_results)
analyser.plot_learning_curves(plot_interval=5, show=False)
