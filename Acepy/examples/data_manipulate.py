# split instance
import numpy as np
from acepy.data_manipulate import split, split_features

X = np.random.rand(10, 2)  # 10 instances with 2 features
y = [0] * 5 + [1] * 5
print('X:  ', X)
print('y:  ', y)

train, test, lab, unlab = split(X=X, y=y, test_ratio=0.5, initial_label_rate=0.5,
                                split_count=1, all_class=True)

print(train)
print(test)
print(lab)
print(unlab)

train, test, lab, unlab = split_features(feature_matrix=X, test_ratio=0.5, missing_rate=0.5,
                                         split_count=1)

print(train)
print(test)
print(lab)
print(unlab)
