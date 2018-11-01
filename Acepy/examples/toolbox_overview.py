import copy

from sklearn.datasets import load_iris

from acepy.utils.toolbox import ToolBox

X, y = load_iris(return_X_y=True)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)

# Use the default Logistic Regression classifier
model = acebox.get_default_model()

# The cost budget is 50 times querying.
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

# Use pre-defined strategy, The data matrix is a reference which will not use additional memory
QBCStrategy = acebox.get_query_strategy(strategy_name='QBC')

QBC_result = []
for round in range(10):
    # Get the data split of one fold experiment
    train_idx, test_idx, Lind, Uind = acebox.get_split(round)

    # Get intermediate results saver for one fold experiment
    saver = acebox.get_stateio(round)

    while not stopping_criterion.is_stop():
        # Select a subset of Uind according to the query strategy
        select_ind = QBCStrategy.select(Lind, Uind, model=model, batch_size=1)
        Lind.update(select_ind)
        Uind.difference_update(select_ind)

        # Update model and calc performance according to the model you are using
        model.fit(X=X[Lind.index, :], y=y[Lind.index])
        pred = model.predict(X[test_idx, :])
        accuracy = sum(pred == y[test_idx]) / len(test_idx)

        # Save intermediate results to file
        st = acebox.State(select_index=select_ind, performance=accuracy)
        saver.add_state(st)
        saver.save()

        # Passing the current progress to stopping criterion object
        stopping_criterion.update_information(saver)
    # Reset the progress in stopping criterion object
    stopping_criterion.reset()
    QBC_result.append(copy.deepcopy(saver))

analyser = acebox.get_experiment_analyser(x_axis='num_of_queries')
analyser.add_method('QBC', QBC_result)
print(analyser)
analyser.plot_line_chart(title='make_classification')
