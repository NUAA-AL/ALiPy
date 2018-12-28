from sklearn.datasets import load_iris
from acepy import ToolBox

X, y = load_iris(return_X_y=True)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# get tools
tr, te, lab, unlab = acebox.split_AL()
tr0, te0, lab0, unlab0 = acebox.get_split(round=0)
oracle = acebox.get_clean_oracle()
saver = acebox.get_stateio(round=0)
repo = acebox.get_repository(round=0)
rand_strategy = acebox.get_query_strategy(strategy_name="QueryRandom")
perf = acebox.calc_performance_metric(y_true=[1], y_pred=[1], performance_metric='accuracy_score')
model = acebox.get_default_model()
sc = acebox.get_stopping_criterion(stopping_criteria='num_of_queries', value=50)
analyser = acebox.get_experiment_analyser(x_axis='num_of_queries')
acethread = acebox.get_ace_threading()

# data struct defined in acepy
ind = acebox.IndexCollection([1, 2, 3])
m_ind = acebox.MultiLabelIndexCollection([(1, 0), (2, )])
st = acebox.State(select_index=[1], performance=perf)

# io
acebox.save('./acebox.pkl')
acebox = ToolBox.load(path='./acebox.pkl')
