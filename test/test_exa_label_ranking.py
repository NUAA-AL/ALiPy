import numpy as np

from alipy.query_strategy.multi_label import LabelRankingModel

X = np.random.rand(100, 10)   # 100 instances
y = np.random.randint(2, size=(100, 4))

labrank = LabelRankingModel()
labrank.fit(X, y)

# it will return 2 values, the first is the decision values, the second is the predicted labels.
pres, label = labrank.predict(X)
print(label)

# label ranking model also support incremental training
labrank.fit(X, y, is_incremental=True)
