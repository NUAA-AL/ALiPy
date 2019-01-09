import numpy as np 
import copy
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_multilabel_classification
from sklearn.ensemble import RandomForestClassifier

from alipy import ToolBox
from alipy.index.multi_label_tools import get_Xy_in_multilabel, check_index_multilabel
from alipy.query_strategy.cost_sensitive import QueryCostSensitiveHALC, QueryCostSensitivePerformance, QueryCostSensitiveRandom
from alipy.query_strategy.cost_sensitive import hierarchical_multilabel_mark
from alipy.metrics.performance import type_of_target

X, y = make_multilabel_classification(n_samples=2000, n_features=20, n_classes=5,
                                   n_labels=3, length=50, allow_unlabeled=True,
                                   sparse=False, return_indicator='dense',
                                   return_distributions=False,
                                   random_state=None)
y[y == 0] = -1

cost = [1, 3, 3, 7, 10]

label_tree = np.zeros((5,5),dtype=np.int)
label_tree[0, 1] = 1
label_tree[0, 2] = 1
label_tree[1, 3] = 1
label_tree[2, 4] = 1

alibox = ToolBox(X=X, y=y, query_type='PartLabels')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = RandomForestClassifier()
# The cost budget is 20 times querying
stopping_criterion = alibox.get_stopping_criterion('num_of_queries', 10)
# The budget of query
budget = 40

performance_result = []
halc_result = []
random_result = []

def main_loop(alibox, strategy, round):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = strategy.select(label_ind, unlab_ind, cost=cost, budget=budget)
        
        select_ind = hierarchical_multilabel_mark(select_ind, label_ind,label_tree, y)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)
            
        # Update model and calc performance according to the model you are using
        X_tr, y_tr, _ = get_Xy_in_multilabel(label_ind, X=X, y=y)
        model.fit(X_tr, y_tr)
        pred = model.predict(X[test_idx, :])
        pred[pred == 0] = 1

        performance = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred, performance_metric='hamming_loss')

        # Save intermediate results to file
        st = alibox.State(select_index=select_ind.index, performance=performance, cost=budget)
        saver.add_state(st)
        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    return saver

for round in range(5):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Use pre-defined strategy
    random = QueryCostSensitiveRandom(X,y)
    perf = QueryCostSensitivePerformance(X, y)
    halc = QueryCostSensitiveHALC(X, y,label_tree=label_tree)

    random_result.append(copy.deepcopy(main_loop(alibox, random, round)))
    performance_result.append(copy.deepcopy(main_loop(alibox, perf, round)))
    halc_result.append(copy.deepcopy(main_loop(alibox, halc, round)))

analyser = alibox.get_experiment_analyser(x_axis='cost')
analyser.add_method(method_name='random', method_results=random_result)
analyser.add_method(method_name='performance', method_results=performance_result)
analyser.add_method(method_name='HALC', method_results=halc_result)

print(analyser)
analyser.plot_learning_curves(title='Example of cost-sensitive', std_area=False)
