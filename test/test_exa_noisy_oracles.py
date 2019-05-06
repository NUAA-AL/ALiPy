from alipy.toolbox import ToolBox
from alipy.oracle import Oracle, Oracles
from alipy.utils.misc import randperm
from alipy.query_strategy.noisy_oracles import QueryNoisyOraclesCEAL, QueryNoisyOraclesAll, \
    QueryNoisyOraclesIEthresh, QueryNoisyOraclesRandom, get_majority_vote
from sklearn.datasets import make_classification
import copy
import numpy as np

X, y = make_classification(n_samples=800, n_features=20, n_informative=2, n_redundant=2,
                           n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
alibox.split_AL(test_ratio=0.3, initial_label_rate=0.15, split_count=10)

# Use the default Logistic Regression classifier
model = alibox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = alibox.get_stopping_criterion('cost_limit', 30)

# initialize noisy oracles with different noise level
n_samples = len(y)
y1 = y.copy()
y2 = y.copy()
y3 = y.copy()
y4 = y.copy()
y5 = y.copy()
perms = randperm(n_samples-1)
y1[perms[0:round(n_samples*0.1)]] = 1-y1[perms[0:round(n_samples*0.1)]]
perms = randperm(n_samples-1)
y2[perms[0:round(n_samples*0.2)]] = 1-y2[perms[0:round(n_samples*0.2)]]
perms = randperm(n_samples-1)
y3[perms[0:round(n_samples*0.3)]] = 1-y3[perms[0:round(n_samples*0.3)]]
perms = randperm(n_samples-1)
y4[perms[0:round(n_samples*0.4)]] = 1-y4[perms[0:round(n_samples*0.4)]]
perms = randperm(n_samples-1)
y5[perms[0:round(n_samples*0.5)]] = 1-y5[perms[0:round(n_samples*0.5)]]
oracle1 = Oracle(labels=y1, cost=np.zeros(y.shape)+1.2)
oracle2 = Oracle(labels=y2, cost=np.zeros(y.shape)+.8)
oracle3 = Oracle(labels=y3, cost=np.zeros(y.shape)+.5)
oracle4 = Oracle(labels=y4, cost=np.zeros(y.shape)+.4)
oracle5 = Oracle(labels=y5, cost=np.zeros(y.shape)+.3)
oracle6 = Oracle(labels=[0]*n_samples, cost=np.zeros(y.shape)+.3)
oracle7 = Oracle(labels=[1]*n_samples, cost=np.zeros(y.shape)+.3)
oracles = Oracles()
oracles.add_oracle(oracle_name='o1', oracle_object=oracle1)
oracles.add_oracle(oracle_name='o2', oracle_object=oracle2)
oracles.add_oracle(oracle_name='o3', oracle_object=oracle3)
oracles.add_oracle(oracle_name='o4', oracle_object=oracle4)
# oracles.add_oracle(oracle_name='o5', oracle_object=oracle5)
oracles.add_oracle(oracle_name='oa0', oracle_object=oracle6)
oracles.add_oracle(oracle_name='oa1', oracle_object=oracle7)

# oracles_list = [oracle1, oracle2]

# def main loop
def al_loop(strategy, alibox, round):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = alibox.get_stateio(round)
    # Get repository to store noisy labels
    repo = alibox.get_repository(round)

    while not stopping_criterion.is_stop():
        # Query
        select_ind, select_ora = strategy.select(label_ind, unlab_ind)
        vote_count, vote_result, cost = get_majority_vote(selected_instance=select_ind, oracles=oracles, names=select_ora)
        repo.update_query(labels=vote_result, indexes=select_ind)

        # update ind
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Train/test
        _, y_lab, indexes_lab = repo.get_training_data()
        model.fit(X=X[indexes_lab], y=y_lab)
        pred = model.predict(X[test_idx])
        perf = alibox.calc_performance_metric(y_true=y[test_idx], y_pred=pred)

        # save
        st = alibox.State(select_index=select_ind, performance=perf, cost=cost)
        saver.add_state(st)

        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    return saver

ceal_result = []
iet_result = []
all_result = []
rand_result = []

for round in range(1):
    train_idx, test_idx, label_ind, unlab_ind = alibox.get_split(round)
    # init strategies
    ceal = QueryNoisyOraclesCEAL(X, y, oracles=oracles, initial_labeled_indexes=label_ind)
    iet = QueryNoisyOraclesIEthresh(X=X, y=y, oracles=oracles, initial_labeled_indexes=label_ind)
    all = QueryNoisyOraclesAll(X=X, y=y, oracles=oracles)
    rand = QueryNoisyOraclesRandom(X=X, y=y, oracles=oracles)

    ceal_result.append(copy.deepcopy(al_loop(ceal, alibox, round)))
    iet_result.append(copy.deepcopy(al_loop(iet, alibox, round)))
    all_result.append(copy.deepcopy(al_loop(all, alibox, round)))
    rand_result.append(copy.deepcopy(al_loop(rand, alibox, round)))

print(oracles.full_history())
analyser = alibox.get_experiment_analyser(x_axis='cost')
analyser.add_method(method_results=ceal_result, method_name='ceal')
analyser.add_method(method_results=iet_result, method_name='iet')
analyser.add_method(method_results=all_result, method_name='all')
analyser.add_method(method_results=rand_result, method_name='rand')
analyser.plot_learning_curves(show=False)
