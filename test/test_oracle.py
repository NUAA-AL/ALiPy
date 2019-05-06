"""
Test functions of oracle modules
"""

from __future__ import division

import random

import numpy as np
from sklearn.datasets import load_iris, make_multilabel_classification

from alipy.oracle.oracle import Oracle, OracleQueryMultiLabel, Oracles

X, y = load_iris(return_X_y=True)
X = X[0:100, ]
o = y[0:50]
oracle = Oracle(o)
oracle2 = Oracle(y[51:])


def test_Oracle():
    for i in range(10):
        r = random.randrange(0, 50)
        test, _ = oracle.query_by_index(r)
        assert test == y[r]
    for i in range(10):
        # oracle._add_one_entry(y[50+i], 50+i)
        oracle.add_knowledge(y[50 + i], 50 + i)
        test, _ = oracle.query_by_index(50+i)
        assert test == y[50+i]
    for i in range(5):
        knowl = y[(60+10*i): (70+10*i)] 
        label = [(60+10*i+j) for j in range(10)]
        assert len(knowl) == len(label)
        oracle.add_knowledge(knowl, label)
    for i in range(50):
        test, _ = oracle.query_by_index(60+i)
        assert test == y[60+i]


Mx, My = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2, length=50, 
                                  allow_unlabeled=True, sparse=False, return_indicator='dense', 
                                 return_distributions=False, random_state=None)

OQM = OracleQueryMultiLabel(My[0:20])


def test_OracleQueryMultiLabel():
    for i in range(10):
        r = random.randrange(0, 20)
        test, _ = OQM.query_by_index((r,))
        assert (test == np.array(My[r])).all()

    for i in range(20):
        OQM._add_one_entry(My[20+i], 20+i)
        test, _ = OQM.query_by_index((20+i,))
        assert (test == np.array(My[20+i])).all()

    for i in range(10):
        r = random.randrange(0, 20)
        for j in range(5):
            test, _ = OQM.query_by_index((r, (j)))
            assert test == np.array(My[r, j])

    for i, j in zip(range(10), range(10, 0, -1)):
        r = random.randrange(0, 20)
        for k in range(5):
            test, _ = OQM.query_by_index([(i, k), (j, k)])     
            assert test[0] == My[i, k]
            assert test[1] == My[j, k]

    for i, j in zip(range(10), range(10, 0, -1)):
        r = random.randrange(0, 20)
        for k in range(5):
            test, _ = OQM.query_by_index([(i, k), (j,)])               
            assert test[0] == My[i, k]
            assert (test[1] == My[j]).all()
    

multi_oracles = Oracles()
def test_Oracles():
    multi_oracles.add_oracle('oracle1', oracle)
    multi_oracles.add_oracle('oracle2', oracle2)

    result = multi_oracles.query_from([1], oracle_name='oracle1')
    multi_oracles.query_from_s([1])
    multi_oracles.query_from_s([1,2], oracles_name=['oracle2', 'oracle1'])
    print(multi_oracles.full_history())
    print(multi_oracles.cost_inall)

    '''
    +---------+-----------------+-----------------+--------------------+--------+
    | oracles |        0        |        1        |         2          | in all |
    +---------+-----------------+-----------------+--------------------+--------+
    | oracle1 | query_index:[1] | query_index:[1] | query_index:[1, 2] |   3    |
    |         |   response:[0]  |   response:[0]  |  response:[0, 0]   |        |
    |         |     cost:[1]    |     cost:[1]    |    cost:[1, 1]     |        |
    | oracle2 |        \        | query_index:[1] | query_index:[1, 2] |   2    |
    |         |                 |   response:[1]  |  response:[1, 1]   |        |
    |         |                 |     cost:[1]    |    cost:[1, 1]     |        |
    +---------+-----------------+-----------------+--------------------+--------+
    7
    '''


test_Oracles()