from sklearn.datasets import load_iris
from acepy.toolbox import ToolBox

X, y = load_iris(return_X_y=True)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)
model = acebox.get_default_model()
train_idx, test_idx, Lind, Uind = acebox.get_split(0)
# -------------Initialize---------------
# initilize a strategy object by ToolBox
QBCStrategy = acebox.get_query_strategy(strategy_name='QueryInstanceQBC')

# import the acepy.query_strategy directly
from acepy.query_strategy import QueryInstanceQBC, QueryInstanceUncertainty

uncertainStrategy = QueryInstanceUncertainty(X, y, measure='entropy')
# --------------Select----------------
# select the unlabeled data to query
model.fit(X[Lind.index], y[Lind.index])
select_ind = uncertainStrategy.select(Lind, Lind, batch_size=1, model=model)
print(select_ind)

# Use the default logistic regression model to choose the instances
select_ind = uncertainStrategy.select(Lind, Uind, batch_size=1, model=None)

# Use select_by_prediction_mat() by providing the probabilistic prediction matrix
prob_mat = model.predict_proba(X[Uind.index])
select_ind = QBCStrategy.select_by_prediction_mat(unlabel_index=Uind, predict=prob_mat, batch_size=1)
print(select_ind)

# -------------Implement your own strategy---------------

# or you can also use your own query strategy
# class my_qs_class:
#     	def __init__(self, X=None, y=None, **kwargs):
# 		pass

# 	def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
# 		"""Select instances to query."""
# 		pass
