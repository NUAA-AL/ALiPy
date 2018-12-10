import copy

from sklearn.datasets import make_classification

from acepy.experiment.state import State
from acepy.query_strategy import (QueryInstanceUncertainty,
                                                 QueryRandom,
                                                 )
from acepy.toolbox import ToolBox

X, y = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
    hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
split_count = 5
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

# split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)

# use the default Logistic Regression classifier
model = acebox.get_default_model()

# query 50 times
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

# use pre-defined strategy, The data matrix is a reference which will not use additional memory
randomStrategy = QueryRandom()
uncertainStrategy = QueryInstanceUncertainty(X, y)

oracle = acebox.get_clean_oracle()

random_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    # saver = acebox.StateIO(round)
    saver = acebox.get_stateio(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = randomStrategy.select(Uind)
        label, cost = oracle.query_by_index(select_ind)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # update model and calc performance
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)

        # save intermediate result
        st = State(select_index=select_ind, performance=accuracy, queried_label=label, cost=cost)
        saver.add_state(st)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    random_result.append(copy.deepcopy(saver))

uncertainty_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    # saver = acebox.StateIO(round)
    saver = acebox.get_stateio(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = uncertainStrategy.select(Lind, Uind, model=model)
        label, cost = oracle.query_by_index(select_ind)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # update model and calc performance
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)

        # save intermediate result
        st = State(select_index=select_ind, performance=accuracy, queried_label=label, cost=cost)
        saver.add_state(st)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    uncertainty_result.append(copy.deepcopy(saver))

# 1. The num of query setting
# 2. The cost sensitive setting

def test_list_of_stateio_object1():
    analyser = acebox.get_experiment_analyser()
    analyser.add_method('random', random_result)
    analyser.add_method('uncertainty', uncertainty_result)
    print(analyser)
    analyser.plot_learning_curves(title='make_classification', std_area=False)
    analyser.plot_learning_curves(title='make_classification', std_area=True)

def test_stateio_container1():
    from acepy.experiment.experiment_analyser import StateIOContainer
    rsc = StateIOContainer(method_name='random', method_results=random_result)
    usc = StateIOContainer(method_name='uncertainty', method_results=uncertainty_result)
    analyser = acebox.get_experiment_analyser()
    analyser.add_method('random', rsc)
    analyser.add_method('uncertainty', usc)
    analyser.plot_learning_curves(title='make_classification', std_area=True)

def test_list_of_performance1():
    radom_result = [[0.6, 0.7, 0.8, 0.9], [0.7, 0.7, 0.75, 0.85]]  # 2 folds, 4 queries for each fold.
    uncertainty_result = [[0.7, 0.75, 0.85, 0.9], [0.73, 0.75, 0.88, 0.92]]
    analyser = acebox.get_experiment_analyser()
    analyser.add_method('random', radom_result)
    analyser.add_method('uncertainty', uncertainty_result)
    analyser.plot_learning_curves(title='make_classification', std_area=True)
    analyser.plot_learning_curves(title='make_classification', std_area=True, start_point=0.6)

def test_list_of_stateio_object2():
    analyser = acebox.get_experiment_analyser(x_axis='cost')
    analyser.add_method('random', random_result)
    analyser.add_method('uncertainty', uncertainty_result)
    print(analyser)
    analyser.plot_learning_curves(title='make_classification', std_area=False)
    analyser.plot_learning_curves(title='make_classification', std_area=True)

def test_stateio_container2():
    from acepy.experiment.experiment_analyser import StateIOContainer
    rsc = StateIOContainer(method_name='random', method_results=random_result)
    usc = StateIOContainer(method_name='uncertainty', method_results=uncertainty_result)
    analyser = acebox.get_experiment_analyser(x_axis='cost')
    analyser.add_method('random', rsc)
    analyser.add_method('uncertainty', usc)
    analyser.plot_learning_curves(title='make_classification', std_area=True)

def test_list_of_performance2():
    radom_result = [[(1, 0.6), (2, 0.7), (2, 0.8), (1, 0.9)], [(1, 0.7), (1, 0.7), (1.5, 0.75), (2.5, 0.85)]]  # 2 folds, 4 queries for each fold.
    uncertainty_result = [[(1, 0.7), (2, 0.75), (1, 0.85), (1, 0.9), (1, 0.91)], [(1, 0.73), (2, 0.75), (3, 0.88)]]
    analyser = acebox.get_experiment_analyser(x_axis='cost')
    analyser.add_method('random', radom_result)
    analyser.add_method('uncertainty', uncertainty_result)
    analyser.plot_learning_curves(title='make_classification', std_area=True)
    analyser.plot_learning_curves(title='make_classification', std_area=True, start_point=0.6)

test_stateio_container2()
