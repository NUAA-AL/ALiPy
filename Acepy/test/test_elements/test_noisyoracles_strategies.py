from sklearn.datasets import make_classification
from acepy.toolbox import ToolBox
from acepy.oracle import Oracle, Oracles
import copy
from acepy.query_strategy.noisy_oracles import QueryNoisyOraclesCEAL, QueryNoisyOraclesAll, QueryNoisyOraclesIEthresh, QueryNoisyOraclesRandom, get_majority_vote
import scipy.io as scio

split_count=10

data_root = 'C:\\Code\\AAAI19_exp\\final_exp\\benchmarks_keel.mat'
datasets = scio.loadmat(data_root)
dataname = 'clean1'
data = datasets[dataname]
data = data[0][0]
# print(type(data))
# print(len(data))
# print(data)
X = data[0]
y = data[1].flatten()

# X, y = load_digits(return_X_y=True)
# X, y = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2,
#                            n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.15, class_sep=1.0,
#                            hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = acebox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

oracle1 = Oracle(labels=[1]*len(y))
oracle2 = Oracle(labels=[-1]*len(y))
oracles = Oracles()
oracles.add_oracle(oracle_name='Tom', oracle_object=oracle1)
oracles.add_oracle(oracle_name='Amy', oracle_object=oracle2)
oracles_list = [oracle1, oracle2]

all = QueryNoisyOraclesAll(X=X, y=y, oracles=oracles)
rand = QueryNoisyOraclesRandom(X=X, y=y, oracles=oracles)

for round in range(split_count):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)
    # calc the initial point
    model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)
    saver.set_initial_point(accuracy)
    ceal = QueryNoisyOraclesCEAL(X, y, oracles=oracles, initial_labeled_indexes=label_ind)
    iet = QueryNoisyOraclesIEthresh(X=X, y=y, oracles=oracles, initial_labeled_indexes=label_ind)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind, select_ora = ceal.select(label_ind, unlab_ind)
        print(select_ind)
        print(select_ora)
        select_ind, select_ora = all.select(label_ind, unlab_ind)
        print(select_ind)
        print(select_ora)
        print(get_majority_vote(selected_instance=select_ind, oracles=oracles))
        select_ind, select_ora = iet.select(label_ind, unlab_ind)
        print(select_ind)
        print(select_ora)
        select_ind, select_ora = rand.select(label_ind, unlab_ind)
        print(select_ind)
        print(select_ora)
        break


