"""
Query type related functions
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

import numpy as np

from .base import BaseMultiLabelQuery
from .multi_label import LabelRankingModel
from ..index.index_collections import MultiLabelIndexCollection
from ..index.multi_label_tools import get_Xy_in_multilabel

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
    To this end, the label matrix you provided can have the following additional information:
    1. -1 means irrelevant.
    2. A positive value means relevant, the larger, the more relevant. (However, do not use 2 which is
    defined as the dummy label)

     Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    References
    ----------
    [1] Huang S J , Chen S , Zhou Z H . Multi-label active
        learning: query type matters[C]// Proceedings of
        the 24th International Joint Conference on
        Artificial Intelligence, pages 946-952, Buenos Aires,
        Argentina, July 25-31, 2015
    """

    def __init__(self, X, y):
        super(QueryTypeAURO, self).__init__(X, y)
        self._lr_model = LabelRankingModel()

    def select(self, label_index, unlabel_index, **kwargs):
        """Select a subset from the unlabeled set, return the selected instance and label.

        Parameters
        ----------
        label_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        Returns
        -------
        selected_ins: int
            The index of selected instance.

        y1, y2: int
            The indexes of selected labels.
        """
        if len(unlabel_index) <= 1:
            return unlabel_index
        unlabel_index = self._check_multi_label_ind(unlabel_index)
        label_index = self._check_multi_label_ind(label_index)

        # select instance with least queries
        W = unlabel_index.get_matrix_mask(mat_shape=self.y.shape, fill_value=1)
        unlab_data, _, data_ind = get_Xy_in_multilabel(index=unlabel_index, X=self.X, y=self.y)
        lab_data, lab_lab, _ = get_Xy_in_multilabel(index=label_index, X=self.X, y=self.y)
        self._lr_model.fit(lab_data, lab_lab)
        pres, labels = self._lr_model.predict(unlab_data)
        selected_ins = np.argmin(np.sum(W, axis=1))

        # last line in pres is the predict value of dummy label
        # select label by calculating the distance between each label with dummy label
        y1 = np.argmax(pres[selected_ins, 0:-1])
        dis = np.abs(pres[selected_ins, :] - pres[selected_ins, -1])
        y2 = np.argmin(dis)

        return selected_ins, y1, y2
