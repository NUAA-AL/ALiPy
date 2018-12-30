import copy
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from acepy.data_manipulate.al_split import split_features
from acepy.query_strategy.query_features import QueryFeatureAFASMC, QueryFeatureRandom, QueryFeatureStability, \
    AFASMC_mc, IterativeSVD_mc
from acepy.index import MultiLabelIndexCollection
from acepy.experiment.stopping_criteria import StoppingCriteria
from acepy.experiment import StateIO, State, ExperimentAnalyser
from acepy.metrics import accuracy_score
from acepy.index import map_whole_index_to_train

# load and split data
X, y = make_classification(n_samples=800, n_features=20, n_informative=2, n_redundant=2,
                           n_repeated=0, n_classes=2, n_clusters_per_class=1, weights=None, flip_y=0.01,
                           hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
tr, te, lab, unlab = split_features(feature_matrix=X, test_ratio=0.3, missing_rate=0.5,
                                    split_count=10, saving_path=None)

# Use the default Logistic Regression classifier
model = LogisticRegression()

# The cost budget is 50 times querying
stopping_criterion = StoppingCriteria('num_of_queries', 50)

AFASMC_result = []
rand_result =[]
Stable_result = []

# AFASMC
for i in range(5):
    train_idx = tr[i]
    test_idx = te[i]
    label_ind = MultiLabelIndexCollection(lab[i], label_size=X.shape[1])
    unlab_ind = MultiLabelIndexCollection(unlab[i], label_size=X.shape[1])
    saver = StateIO(i, train_idx, test_idx, label_ind, unlab_ind)
    strategy = QueryFeatureAFASMC(X=X, y=y, train_idx=train_idx)

    while not stopping_criterion.is_stop():
        # query
        selected_feature = strategy.select(observed_entries=label_ind, unkonwn_entries=unlab_ind)

        # update index
        label_ind.update(selected_feature)
        unlab_ind.difference_update(selected_feature)

        # train/test
        lab_in_train = map_whole_index_to_train(train_idx, label_ind)
        X_mc = AFASMC_mc(X=X[train_idx], y=y[train_idx], omega=lab_in_train)
        model.fit(X_mc, y[train_idx])
        pred = model.predict(X[test_idx])
        perf = accuracy_score(y_true=y[test_idx], y_pred=pred)

        # save
        st = State(select_index=selected_feature, performance=perf)
        saver.add_state(st)
        # saver.save()

        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    AFASMC_result.append(copy.deepcopy(saver))

SVD_mc = IterativeSVD_mc(rank=4)
# Stablility
for i in range(5):
    train_idx = tr[i]
    test_idx = te[i]
    label_ind = MultiLabelIndexCollection(lab[i], label_size=X.shape[1])
    unlab_ind = MultiLabelIndexCollection(unlab[i], label_size=X.shape[1])
    saver = StateIO(i, train_idx, test_idx, label_ind, unlab_ind)
    strategy = QueryFeatureStability(X=X, y=y, train_idx=train_idx, rank_arr=[4, 6, 8])

    while not stopping_criterion.is_stop():
        # query
        selected_feature = strategy.select(observed_entries=label_ind, unkonwn_entries=unlab_ind)

        # update index
        label_ind.update(selected_feature)
        unlab_ind.difference_update(selected_feature)

        # train/test
        lab_in_train = map_whole_index_to_train(train_idx, label_ind)
        X_mc = SVD_mc.impute(X[train_idx], observed_mask=lab_in_train.get_matrix_mask(mat_shape=(len(train_idx), X.shape[1]), sparse=False))
        model.fit(X_mc, y[train_idx])
        pred = model.predict(X[test_idx])
        perf = accuracy_score(y_true=y[test_idx], y_pred=pred)

        # save
        st = State(select_index=selected_feature, performance=perf)
        saver.add_state(st)

        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    Stable_result.append(copy.deepcopy(saver))

# rand
for i in range(5):
    train_idx = tr[i]
    test_idx = te[i]
    label_ind = MultiLabelIndexCollection(lab[i], label_size=X.shape[1])
    unlab_ind = MultiLabelIndexCollection(unlab[i], label_size=X.shape[1])
    saver = StateIO(i, train_idx, test_idx, label_ind, unlab_ind)
    strategy = QueryFeatureRandom()

    while not stopping_criterion.is_stop():
        # query
        selected_feature = strategy.select(observed_entries=label_ind, unkonwn_entries=unlab_ind)

        # update index
        label_ind.update(selected_feature)
        unlab_ind.difference_update(selected_feature)

        # train/test
        lab_in_train = map_whole_index_to_train(train_idx, label_ind)
        X_mc = SVD_mc.impute(X[train_idx], observed_mask=lab_in_train.get_matrix_mask(mat_shape=(len(train_idx), X.shape[1]), sparse=False))
        model.fit(X_mc, y[train_idx])
        pred = model.predict(X[test_idx])
        perf = accuracy_score(y_true=y[test_idx], y_pred=pred)

        # save
        st = State(select_index=selected_feature, performance=perf)
        saver.add_state(st)

        stopping_criterion.update_information(saver)

    stopping_criterion.reset()
    rand_result.append(copy.deepcopy(saver))

analyser = ExperimentAnalyser()
analyser.add_method(method_results=AFASMC_result, method_name='AFASMC')
analyser.add_method(method_results=Stable_result, method_name='Stability')
analyser.add_method(method_results=rand_result, method_name='Random')
print(analyser)
analyser.plot_learning_curves()



