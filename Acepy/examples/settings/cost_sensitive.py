import sys
sys.path.append(r'D:\Al_tool\Acepy')
import numpy as np 
# labels = [0, 1 , 0, 3]
# print(np.unique(labels))
# cost = [2, 1, 2, 5]
# from acepy.oracle import Oracle
# oracle = Oracle(labels=labels, cost=cost)

# labels, cost = oracle.query_by_index(indexes=np.unique(labels))

# print(labels)
# print(cost)
# from acepy.experiment import State
# st = State(select_index=select_ind, performance=accuracy, cost=cost)

# radom_result = [[(1, 0.6), (2, 0.7), (2, 0.8), (1, 0.9)],
#                 [(1, 0.7), (1, 0.7), (1.5, 0.75), (2.5, 0.85)]]  # 2 folds, 4 queries for each fold.
# uncertainty_result = [saver1, saver2]  # each State object in the saver must have the 'cost' entry.
# from acepy.experiment import ExperimentAnalyser

# analyser = ExperimentAnalyser(x_axis='cost')
# analyser.add_method('random', radom_result)
# analyser.add_method('uncertainty', uncertainty_result)


import copy
from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier

from acepy import ToolBox
from acepy.index.multi_label_tools import get_Xy_in_multilabel, check_index_multilabel
from acepy.query_strategy.cost_sensitive import QueryCostSensitiveHALC, QueryCostSensitivePerformance, QueryCostSensitiveRandom
from acepy.query_strategy.cost_sensitive import hierarchical_multilabel_mark


X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5,
                                   n_labels=3, length=50, allow_unlabeled=True,
                                   sparse=False, return_indicator='dense',
                                   return_distributions=False,
                                   random_state=None)
y[y == 0] = -1
cost = [1, 3, 3, 7, 10]
label_tree = np.zeros((5,5),dtype=np.int)
label_tree[0, ] = 1
label_tree[1, 3] = 1
label_tree[2, 4] = 1

acebox = ToolBox(X=X, y=y, query_type='PartLabels')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = RandomForestClassifier()

# The cost budget is 20 times querying
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 20)

# The budget of query
budget = 20

performance_result = []
halc_result = []
random_result = []




def main_loop(acebox, strategy, round):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = strategy.select(label_ind, unlab_ind, cost=cost, budget=budget)
        select_ind = hierarchical_multilabel_mark(select_ind, label_tree, y)
        # print('select_ind type', type(select_ind))
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        X_tr, y_tr, _ = get_Xy_in_multilabel(label_ind, X=X, y=y)
        model.fit(X_tr, y_tr)
        # model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = acebox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='hamming_loss')

        # Save intermediate results to file
        # check_index_multilabel(select_ind)
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    return saver

for round in range(5):
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)

    # Use pre-defined strategy
    random = QueryCostSensitiveRandom(X,y)
    perf = QueryCostSensitivePerformance(X, y)
    halc = QueryCostSensitiveHALC(X, y,label_tree=label_tree)


    performance_result.append(copy.deepcopy(main_loop(acebox, random, round)))
    random_result.append(copy.deepcopy(main_loop(acebox, perf, round)))
    halc_result.append(copy.deepcopy(main_loop(acebox, halc, round)))



analyser = acebox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='random', method_results=performance_result)
analyser.add_method(method_name='performance', method_results=random_result)
analyser.add_method(method_name='HALC', method_results=halc_result)

print(analyser)
analyser.plot_learning_curves(title='Example of acepy', std_area=False)