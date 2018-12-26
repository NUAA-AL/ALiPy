import random
import os
import unittest

import sys
sys.path.append(r'C:\Users\31236\Desktop\al_tools\acepy')

from numpy.testing import assert_array_equal
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression

from acepy.index import IndexCollection
from acepy.query_strategy.multi_label import MaximumLossReductionMaximalConfidence, AdaptiveActiveLearning


dataset_filepath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'yeast_train.svm')

X, y = load_svmlight_file(dataset_filepath, multilabel=True)
X = X.todense().tolist()
y = MultiLabelBinarizer().fit_transform(y).tolist()
quota = 10
X = np.array(X)
y = np.array(y)

# MMC = MaximumLossReductionMaximalConfidence(X, y, random_state=1126)

# label_index = IndexCollection([0, 1, 2, 3, 4])
# unlabel_index = IndexCollection(list(range(5,1500)))
# for i in range(10):
#     selected_index = MMC.select(label_index, unlabel_index)
#     print(selected_index)
#     label_index.add(selected_index)
#     unlabel_index.discard(selected_index)


# [117, 655, 1350, 909, 1003, 1116, 546, 1055, 165, 1441]




AAL = AdaptiveActiveLearning(X, y, base_clf=LogisticRegression(), random_state=1126)

label_index = IndexCollection([0, 1, 2, 3, 4])
unlabel_index = IndexCollection(list(range(5,1500)))
for i in range(10):
    selected_index = AAL.select(label_index, unlabel_index)
    print(selected_index)
    label_index.add(selected_index)
    unlabel_index.discard(selected_index)

[594, 827, 1128, 419, 1223, 484, 96, 833, 37, 367]