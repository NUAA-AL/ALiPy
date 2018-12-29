import copy
from sklearn.datasets import make_classification
from acepy import ToolBox
from acepy.query_strategy.query_labels import QueryInstanceGraphDensity, QueryInstanceQBC, \
    QueryInstanceQUIRE, QueryRandom, QueryInstanceUncertainty, QureyExpectedErrorReduction, QueryInstanceLAL

X, y = make_classification(n_samples=500, n_features=20, n_informative=2, n_redundant=2,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
    hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = acebox.get_default_model()

# The cost budget is 50 times querying
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)


QBC_result = []

def main_loop(acebox, strategy, round):
    # Get the data split of one fold experiment
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)
    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)
    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        # Passing model=None to use the default model for evaluating the committees' disagreement
        select_ind = strategy.select(label_ind, unlab_ind, batch_size=1)
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

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    return saver


unc_result = []
qbc_result = []
eer_result = []
quire_result = []
density_result = []
bmdr_result = []
spal_result = []
lal_result = []

_I_have_installed_the_cvxpy = True

for round in range(5):
    train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round)

    # Use pre-defined strategy
    unc = QueryInstanceUncertainty(X, y)
    qbc = QueryInstanceQBC(X, y)
    eer = QureyExpectedErrorReduction(X, y)
    quire = QueryInstanceQUIRE(X, y, train_idx)
    density = QueryInstanceGraphDensity(X, y ,train_idx)
    lal = QueryInstanceLAL(X, y, cls_est=10, train_slt=False)
    lal.download_data()
    lal.train_selector_from_file(reg_est=30, reg_depth=5)

    unc_result.append(copy.deepcopy(main_loop(acebox, unc, round)))
    qbc_result.append(copy.deepcopy(main_loop(acebox, qbc, round)))
    eer_result.append(copy.deepcopy(main_loop(acebox, eer, round)))
    quire_result.append(copy.deepcopy(main_loop(acebox, quire, round)))
    density_result.append(copy.deepcopy(main_loop(acebox, density, round)))
    lal_result.append(copy.deepcopy(main_loop(acebox, lal, round)))

    if _I_have_installed_the_cvxpy:
        from acepy.query_strategy.query_labels import QueryInstanceBMDR, QueryInstanceSPAL
        bmdr = QueryInstanceBMDR(X, y, kernel='linear')
        spal = QueryInstanceSPAL(X, y, kernel='linear')

        bmdr_result.append(copy.deepcopy(main_loop(acebox, bmdr, round)))
        spal_result.append(copy.deepcopy(main_loop(acebox, spal, round)))


analyser = acebox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method(method_name='QBC', method_results=qbc_result)
analyser.add_method(method_name='Unc', method_results=unc_result)
analyser.add_method(method_name='EER', method_results=eer_result)
analyser.add_method(method_name='QUIRE', method_results=quire_result)
analyser.add_method(method_name='Density', method_results=density_result)
analyser.add_method(method_name='LAL', method_results=lal_result)
if _I_have_installed_the_cvxpy:
    analyser.add_method(method_name='BMDR', method_results=bmdr_result)
    analyser.add_method(method_name='SPAL', method_results=spal_result)
print(analyser)
analyser.plot_learning_curves(title='Example of acepy', std_area=False)
