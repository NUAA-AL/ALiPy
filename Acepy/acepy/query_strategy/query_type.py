"""
Query type related functions
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

import numpy as np

from .base import BaseMultiLabelQuery


def check_query_type(type):
    """Check the query type.

    Only the following query types are allowed:
    AllowedType:
        AllLabels: Query all _labels of an instance
        PartLabels: Query part of labels of an instance (Only available in multi-label setting)
        Features: Query unlab_features of instances
    NotImplementedQueryType
        Relations: Query relations between two object
        Examples: Query examples given constrains


    AllLabels: query all labels of an selected instance.
        Support scene: binary classification, multi-class classification, multi-label classification, regression

    Partlabels: query part of labels of an instance.
        Support scene: multi-label classification

    Features: query part of features of an instance.
        Support scene: missing features

    Parameters
    ----------
    type: str
        query type.

    Returns
    -------
    result: bool
        if query type in ['AllLabels', 'PartLabels', 'Features'],return True.
    """
    assert (isinstance(type, str))
    QueryType = ['AllLabels', 'PartLabels', 'Features']
    NotImplementedQueryType = ['Relations', 'Examples']
    if type in QueryType:
        return True
    else:
        return False


class QueryTypeAURO(BaseMultiLabelQuery):
    """AURO select one instance and its 2 labels to query which one is more relevant.

    The query type of this method is different with the normal active learning
    algorithms that always query labels.

     Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    References
    ----------
    [1] Huang, S.; Jin, R.; and Zhou, Z. 2014. Active learning by
        querying informative and representative examples. IEEE
        Transactions on Pattern Analysis and Machine Intelligence
        36(10):1936–1949
    """

    def __init__(self, X, y, **kwargs):
        super(QueryTypeAURO, self).__init__(X, y)
