from __future__ import division

import pytest
from sklearn.datasets import load_iris, make_classification

import acepy
from acepy.query_strategy.query_strategy import (QueryInstanceUncertainty,
                                                 QueryInstanceQBC)

X, y = load_iris(return_X_y=True)

split_count = 5
acebox = acepy.ToolBox(X=X, y=y, query_type='AllLabels', saving_path=None)

# split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=split_count)

# use the default Logistic Regression classifier
model = acebox.get_default_model()

# query 50 times
stopping_criterion = acebox.get_stopping_criterion('num_of_queries', 50)

# use pre-defined strategy, The data matrix is a reference which will not use additional memory

def test_uncertainty():
    # multi class
    least_confident = QueryInstanceUncertainty(X, y, measure='least_confident')
    margin = QueryInstanceUncertainty(X, y, measure='margin')
    entropy = QueryInstanceUncertainty(X, y, measure='entropy')

    # get split
    train_idx, test_idx, Lind, Uind = acebox.get_split(0)

    # query
    select_ind1 = least_confident.select(Lind, Uind, model=None)
    select_ind2 = margin.select(Lind, Uind, model=None)
    select_ind3 = entropy.select(Lind, Uind, model=None)
    # assert select_ind1[0] == select_ind2[0] == select_ind3[0]

    # binary
    X_2, y_2 = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2,
                                   n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01,
                                   class_sep=1.0,
                                   hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    distance_to_boundary = QueryInstanceUncertainty(X_2, y_2, measure='distance_to_boundary')
    least_confident = QueryInstanceUncertainty(X_2, y_2, measure='least_confident')
    margin = QueryInstanceUncertainty(X_2, y_2, measure='margin')
    entropy = QueryInstanceUncertainty(X_2, y_2, measure='entropy')

    model.fit(X=X_2[Lind.index, :], y=y_2[Lind.index])

    # query
    select_ind1 = least_confident.select(Lind, Uind, model=None)
    select_ind2 = margin.select(Lind, Uind, model=None)
    select_ind3 = entropy.select(Lind, Uind, model=None)
    select_ind4 = distance_to_boundary.select(Lind, Uind, model=None)
    assert select_ind1[0] == select_ind2[0] == select_ind3[0] == select_ind4[0]

    # select by mat
    select_ind1 = least_confident.select_by_prediction_mat(unlabel_index=Uind,
                                                           predict=model.predict_proba(X_2[Uind.index]))
    select_ind2 = margin.select_by_prediction_mat(unlabel_index=Uind, predict=model.predict_proba(X_2[Uind.index]))
    select_ind3 = entropy.select_by_prediction_mat(unlabel_index=Uind, predict=model.predict_proba(X_2[Uind.index]))
    select_ind4 = distance_to_boundary.select_by_prediction_mat(unlabel_index=Uind,
                                                                predict=model.decision_function(X_2[Uind.index]))
    assert select_ind1[0] == select_ind2[0] == select_ind3[0] == select_ind4[0]


def test_QBC():
    vote_entropy = QueryInstanceQBC(X, y, disagreement='vote_entropy')
    KL_divergence = QueryInstanceQBC(X, y, disagreement='KL_divergence')

    # get split
    train_idx, test_idx, Lind, Uind = acebox.get_split(0)
    model.fit(X[train_idx], y[train_idx])

    # query
    select_ind1 = vote_entropy.select(Lind, Uind, model=None)
    select_ind2 = KL_divergence.select(Lind, Uind, model=None)

    vote_entropy.select_by_prediction_mat(Uind, [model.predict(X[Uind.index]), model.predict(X[Uind.index])])
    KL_divergence.select_by_prediction_mat(Uind,
                                           [model.predict_proba(X[Uind.index]), model.predict_proba(X[Uind.index])])
    with pytest.raises(ValueError, match=r'Two or more committees are expected.*'):
        vote_entropy.select_by_prediction_mat(Uind, [model.predict_proba(X[Uind.index])])
