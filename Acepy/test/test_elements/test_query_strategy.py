from __future__ import division
import pytest
import numpy as np
from sklearn.datasets import make_classification
from acepy.query_strategy import query_strategy, sota_strategy

import copy
from sklearn.datasets import load_iris, make_classification

from acepy.experiment.state import State
from acepy.utils.toolbox import ToolBox

from acepy.query_strategy.query_strategy import (QueryInstanceQBC,
                                           QueryInstanceUncertainty,
                                           QueryRandom,
                                           QureyExpectedErrorReduction)
from acepy.query_strategy.sota_strategy import QueryInstanceQUIRE, QueryInstanceGraphDensity
from acepy.index.index_collections import IndexCollection


X, y = load_iris(return_X_y=True)

split_count = 5
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

# split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)

# use the default Logistic Regression classifier
model = acebox.default_model()

# query 50 times
stopping_criterion = acebox.stopping_criterion('num_of_queries', 50)

# use pre-defined strategy, The data matrix is a reference which will not use additional memory
QBCStrategy = QueryInstanceQBC(X, y)
randomStrategy = QueryRandom()
uncertainStrategy = QueryInstanceUncertainty(X, y)
QUIREStrategy = QueryInstanceQUIRE(X, y)
EER = QureyExpectedErrorReduction(X, y)

def test_():
    pass