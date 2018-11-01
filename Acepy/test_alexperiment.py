from sklearn.datasets import load_iris, make_classification
from acepy.experiment.state import State
from acepy.utils.toolbox import ToolBox
from acepy.query_strategy.query_strategy import QueryInstanceUncertainty,QureyExpectedErrorReduction

from acepy.experiment.al_experiment import AlExperiment


X, y = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2, 
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, 
    hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

al = AlExperiment(X, y)
al.split_AL()
user_query = QureyExpectedErrorReduction
print(callable(user_query))
if callable(user_query):
    print("callable")
    al.set_query_strategy(user_query, strategyname='QueryEER')
else:
    al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='entropy')

# al.start_query(multi_thread=False)
al.start_query()
al.get_experiment_result()
