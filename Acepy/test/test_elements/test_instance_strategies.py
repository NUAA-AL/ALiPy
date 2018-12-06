from sklearn.datasets import load_digits
from sklearn.datasets import make_classification
from acepy.toolbox import ToolBox
import copy
from acepy.query_strategy.query_labels import QueryInstanceBMDR, QueryInstanceSPAL, QueryInstanceLAL,\
    QueryInstanceUncertainty, QueryRandom
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

# train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round=0)
#
# bmdr = QueryInstanceBMDR(X, y, kernel='linear')
# select = bmdr.select(label_ind, unlab_ind)
# print(select)
#
# spal = QueryInstanceSPAL(X, y, kernel='linear')
# select = spal.select(label_ind, unlab_ind)
# print(select)
#
# lal = QueryInstanceLAL(X, y, mode='LAL_iterative', train_slt=False)
# # lal.download_data()
# lal.train_selector_from_file(reg_est=10, reg_depth=3, feat=6)
# select = lal.select(label_ind, unlab_ind, batch_size=5)
# print(select)

# Use the default Logistic Regression classifier
model = acebox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

# Use pre-defined strategy
bmdr = QueryInstanceBMDR(X, y, kernel='linear')
spal = QueryInstanceSPAL(X, y, kernel='linear', lambda_init=0.1)
lal = QueryInstanceLAL(X, y, mode='LAL_iterative', cls_est=10, train_slt=False)
lal.train_selector_from_file(reg_est=10, reg_depth=3, feat=6)

bmdr_result = []
spal_result = []
lal_result = []

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

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = bmdr.select(label_ind, unlab_ind, model=None, batch_size=5)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = acebox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    bmdr_result.append(copy.deepcopy(saver))

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

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = spal.select(label_ind, unlab_ind, model=None, batch_size=5)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = acebox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    spal_result.append(copy.deepcopy(saver))

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

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = lal.select(label_ind, unlab_ind, batch_size=5)
        label_ind.update(select_ind)
        unlab_ind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[label_ind.index, :], y=y[label_ind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = acebox.calc_performance_metric(y_true=y[test_idx],
                                                  y_pred=pred,
                                                  performance_metric='accuracy_score')

        # Save intermediate results to file
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    lal_result.append(copy.deepcopy(saver))

randomStrategy = QueryRandom()
uncertainStrategy = QueryInstanceUncertainty(X, y)
random_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.get_stateio(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = randomStrategy.select(Uind, batch_size=5)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # update model and calc performance
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)

        # save intermediate result
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    random_result.append(copy.deepcopy(saver))

uncertainty_result = []
for round in range(split_count):
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)
    saver = acebox.get_stateio(round)

    # calc the initial point
    model.fit(X=X[Lind.index, :], y=y[Lind.index])
    pred = model.predict(X[test_idx, :])
    accuracy = sum(pred == y[test_idx]) / len(test_idx)

    saver.set_initial_point(accuracy)
    while not stopping_criterion.is_stop():
        select_ind = uncertainStrategy.select(Lind, Uind, model=model, batch_size=5)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # update model and calc performance
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)

        # save intermediate result
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # update stopping_criteria
        stopping_criterion.update_information(saver)
    stopping_criterion.reset()
    uncertainty_result.append(copy.deepcopy(saver))


analyser = acebox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='spal', method_results=spal_result)
analyser.add_method(method_name='bmdr', method_results=bmdr_result)
analyser.add_method(method_name='lal', method_results=lal_result)
# analyser.add_method(method_name='unc', method_results=uncertainty_result)
analyser.add_method(method_name='rand', method_results=random_result)
print(analyser)
analyser.plot_learning_curves(title='Example of AL', std_area=False)
