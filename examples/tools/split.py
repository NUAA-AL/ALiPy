import numpy as np
# split instance
X = np.random.rand(10, 10)  # 10 instances with 10 features
y = [0] * 5 + [1] * 5
from alipy.data_manipulate import split
train, test, lab, unlab = split(X=X, y=y, test_ratio=0.5, initial_label_rate=0.5,
                                split_count=1, all_class=True, saving_path='.')
print(train, test, lab, unlab)


# split multi_label
from alipy.data_manipulate import split_multi_label
# 3 instances with 3 labels.
mult_y = [[1, 1, 1], [0, 1, 1], [0, 1, 0]]  
train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
    y=mult_y, split_count=1, all_class=False,
    test_ratio=0.3, initial_label_rate=0.5,
    saving_path=None
)
print(train_idx)
print(test_idx)
print(label_idx)
print(unlabel_idx)


# split features
from alipy.data_manipulate import split_features
X = np.random.rand(10, 2)  # 10 instances with 2 features
train, test, lab, unlab = split_features(feature_matrix=X, test_ratio=0.5, missing_rate=0.5,
                                         split_count=1)
print(train, test, lab, unlab)
