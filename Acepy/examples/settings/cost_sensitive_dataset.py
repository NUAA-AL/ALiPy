

labels = [0, 1 ,0]
cost = [2, 1, 2]
from acepy.oracle import Oracle
oracle = Oracle(labels=labels, cost=cost)

labels, cost = oracle.query_by_index(indexes=[1])

from acepy.experiment import State
st = State(select_index=select_ind, performance=accuracy, cost=cost)

radom_result = [[(1, 0.6), (2, 0.7), (2, 0.8), (1, 0.9)],
                [(1, 0.7), (1, 0.7), (1.5, 0.75), (2.5, 0.85)]]  # 2 folds, 4 queries for each fold.
uncertainty_result = [saver1, saver2]  # each State object in the saver must have the 'cost' entry.
from acepy.experiment import ExperimentAnalyser

analyser = ExperimentAnalyser(x_axis='cost')
analyser.add_method('random', radom_result)
analyser.add_method('uncertainty', uncertainty_result)
