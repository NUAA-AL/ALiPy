"""
Query type related functions.
Acepy implements IJCAI'15 Multi-Label Active Learning:
Query Type Matters (AURO) method which queries the relevance
ordering of the 2 selected labels of an instance in multi label setting,
i.e., ask the oracle which of the two labels is more relevant to the instance.

Due to the less attention to this direction, we only implement AURO
for query type. More strategies will be added when new advanced
methods are proposed in the future.
"""

# Authors: Ying-Peng Tang
# License: BSD 3 clause

import numpy as np

from .base import BaseMultiLabelQuery
from .multi_label import LabelRankingModel
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

    def select(self, label_index, unlabel_index, y_mat=None, **kwargs):
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

        y_mat: array, optional (default=None)
            The label matrix used for model training. Should have the same shape of y.
            Use ground-truth if not given.

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
        if y_mat is None:
            y_mat = self.y

        # select instance with least queries
        W = unlabel_index.get_matrix_mask(mat_shape=self.y.shape, fill_value=1, sparse=False)
        unlab_ins_ind = np.nonzero(np.sum(W, axis=1) > 1)[0]
        unlab_data = self.X[unlab_ins_ind]
        unlab_mask = W[unlab_ins_ind]
        lab_data, lab_lab, _ = get_Xy_in_multilabel(index=label_index, X=self.X, y=y_mat)
        self._lr_model.fit(lab_data, lab_lab)
        pres, labels = self._lr_model.predict(unlab_data)
        selected_ins = np.argmax(np.sum(unlab_mask, axis=1))

        # map index from whole dataset to the unlabeled data
        min_val = pres.min()
        if min_val > 0:
            min_val = -min_val
        pres[:, 0:-1] = pres[:, 0:-1] + min_val*(1-unlab_mask)

        # last line in pres is the predict value of dummy label
        # select label by calculating the distance between each label with dummy label
        y1 = np.argmax(pres[selected_ins, 0:-1])
        dis = np.abs(pres[selected_ins, 0:-1] - pres[selected_ins, -1])
        sort_dis = np.argsort(dis)
        for dis_ind in sort_dis:
            if dis_ind != y1:
                y2 = dis_ind
                break

        return unlab_ins_ind[selected_ins], y1, y2
