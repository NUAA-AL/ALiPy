from sklearn.datasets import load_iris
from alipy import ToolBox

X, y = load_iris(return_X_y=True)
alibox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# get tools
tr, te, lab, unlab = alibox.split_AL()
tr0, te0, lab0, unlab0 = alibox.get_split(round=0)
oracle = alibox.get_clean_oracle()
saver = alibox.get_stateio(round=0)
repo = alibox.get_repository(round=0)
rand_strategy = alibox.get_query_strategy(strategy_name="QueryInstanceRandom")
perf = alibox.calc_performance_metric(y_true=[1], y_pred=[1], performance_metric='accuracy_score')
model = alibox.get_default_model()
sc = alibox.get_stopping_criterion(stopping_criteria='num_of_queries', value=50)
analyser = alibox.get_experiment_analyser(x_axis='num_of_queries')
acethread = alibox.get_ace_threading()

# data struct defined in alipy
ind = alibox.IndexCollection([1, 2, 3])
m_ind = alibox.MultiLabelIndexCollection([(1, 0), (2, )])
st = alibox.State(select_index=[1], performance=perf)

# io
alibox.save()
# al_settings.pkl is the default name. To use another name, please pass a specific file name
# to 'saving_path' parameter when initializing the ToolBox object. (e.g., saving_path='./my_file.pkl')
alibox = ToolBox.load(path='./al_settings.pkl')
