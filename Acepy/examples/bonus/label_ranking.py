from acepy.query_strategy.multi_label import LabelRankingModel
import numpy as np

X = np.random.rand(100, 10)   # 100 instances
y = np.random.randint(2, size=(100, 4))

labrank = LabelRankingModel()
labrank.fit(X, y)
pres, label = labrank.predict(X)
print(label)
