from sklearn.datasets import load_iris
from alipy.experiment import AlExperiment

# Get the data
X, y = load_iris(return_X_y=True)

for strategy in ['QueryInstanceQBC', 'QueryInstanceUncertainty', 'QueryInstanceRandom',
                'QureyExpectedErrorReduction', 'QueryInstanceGraphDensity', 'QueryInstanceQUIRE',
                'QueryInstanceBMDR', 'QueryInstanceSPAL', 'QueryInstanceLAL',
                'QueryExpectedErrorReduction']:
    # init the AlExperiment
    al = AlExperiment(X, y, stopping_criteria='num_of_queries', stopping_value=50)

    # split the data by using split_AL()
    al.split_AL(split_count=5)

    # al.set_query_strategy(strategy=strategy)

    # al.set_performance_metric('accuracy_score')

    # al.start_query(multi_thread=True)

    # or set the data split indexes by input the specific parameters
    from alipy.data_manipulate import split

    train, test, lab, unlab = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.05,
                                    split_count=1)
    al.set_data_split(train_idx=train, test_idx=test, label_idx=lab, unlabel_idx=unlab)

    # set the query strategy
    # using the a pre-defined strategy
    al.set_query_strategy(strategy=strategy)

    # or using your own query strategy
    # class my_qs_class:
    #     	def __init__(self, X=None, y=None, **kwargs):
    # 		pass

    # 	def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
    # 		"""Select instances to query."""
    # 		pass
    # al.set_query_strategy(strategy=my_qs_class(), **kwargs)

    # set the metric for experiment.
    al.set_performance_metric('accuracy_score')

    # by default,run in multi-thread.
    al.start_query(multi_thread=False)
    # or execute sequentially
    # al.start_query(multi_thread=False)

    # get the experiemnt result
    stateIO = al.get_experiment_result()

    # get a brief description of the experiment
    # al.plot_learning_curve(title='Alexperiment result')
