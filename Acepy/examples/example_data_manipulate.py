# split instance
import numpy as np
from acepy.data_manipulate import split
X = np.random.rand(10, 10) # 10 instances with 10 features
y = [0]*5 + [1]*5
train, test, lab, unlab = split(X=X, y=y, test_ratio=0.5, initial_label_rate=0.5,
                                split_count=1, all_class=True)

print(train)
print(test)
print(lab)
print(unlab)
