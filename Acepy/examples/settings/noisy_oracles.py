from acepy.toolbox import ToolBox
from acepy.oracle import Oracle, Oracles
from acepy.query_strategy.noisy_oracles import QueryNoisyOraclesCEAL, QueryNoisyOraclesAll, \
    QueryNoisyOraclesIEthresh, QueryNoisyOraclesRandom, get_majority_vote
from sklearn.datasets import make_classification
import copy

X, y = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2,
                           n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.15, class_sep=1.0,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = acebox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

# initialize noisy oracles
oracle1 = Oracle(labels=[1]*len(y))
oracle2 = Oracle(labels=[-1]*len(y))
oracles = Oracles()
oracles.add_oracle(oracle_name='Tom', oracle_object=oracle1)
oracles.add_oracle(oracle_name='Amy', oracle_object=oracle2)
oracles_list = [oracle1, oracle2]

# def main loop
def al_loop(strategy, acebox, round):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)
    # Get repository to store noisy labels
    repo = acebox.get_repository(round)

    while not stopping_criterion.is_stop():
        # Query
        select_ind, select_ora = strategy.select(label_ind, unlab_ind)
        vote_count, vote_result = get_majority_vote(selected_instance=select_ind, oracles=oracles)
        repo.update_query(labels=vote_result, indexes=select_ind)

        # update ind
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Train/test
        _, y_lab, indexes_lab = repo.get_training_data()
        model.fit(X=X[indexes_lab], y=y_lab)
        pred = model.predict(X[test_idx])
        perf = acebox.calc_performance_metric(y_true=y[test_idx], y_pred=pred)

        # save
        st = acebox.State(select_index=select_ind, performance=perf)
        saver.add_state(st)

        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    return saver

ceal_result = []
iet_result = []
all_result = []
rand_result = []

for round in range(10):
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # init strategies
    ceal = QueryNoisyOraclesCEAL(X, y, oracles=oracles, initial_labeled_indexes=label_ind)
    iet = QueryNoisyOraclesIEthresh(X=X, y=y, oracles=oracles, initial_labeled_indexes=label_ind)
    all = QueryNoisyOraclesAll(X=X, y=y, oracles=oracles)
    rand = QueryNoisyOraclesRandom(X=X, y=y, oracles=oracles)

    ceal_result.append(copy.deepcopy(al_loop(ceal, acebox, round)))
    iet_result.append(copy.deepcopy(al_loop(iet, acebox, round)))
    all_result.append(copy.deepcopy(al_loop(all, acebox, round)))
    rand_result.append(copy.deepcopy(al_loop(rand, acebox, round)))

analyser = acebox.get_experiment_analyser()
analyser.add_method(method_results=ceal_result, method_name='ceal')
analyser.add_method(method_results=iet_result, method_name='iet')
analyser.add_method(method_results=all_result, method_name='all')
analyser.add_method(method_results=rand_result, method_name='rand')
analyser.plot_learning_curves()
