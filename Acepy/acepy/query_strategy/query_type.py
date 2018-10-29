"""
Query type related functions
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

import numpy as np


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

    """
    assert (isinstance(type, str))
    QueryType = ['AllLabels', 'PartLabels', 'Features']
    NotImplementedQueryType = ['Relations', 'Examples']
    if type in QueryType:
        return True
    else:
        return False

