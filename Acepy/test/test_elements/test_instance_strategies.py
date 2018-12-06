from sklearn.datasets import load_iris
from acepy.toolbox import ToolBox
from acepy.query_strategy.query_labels import QueryInstanceBMDR, QueryInstanceSPAL, QueryInstanceLAL

X, y = load_iris(return_X_y=True)
acebox = ToolBox(X=X, y=y, query_type='AllLabels', saving_path='.')

# Split data
acebox.split_AL(test_ratio=0.3, initial_label_rate=0.1, split_count=10)
train_idx, test_idx, label_ind, unlab_ind = acebox.get_split(round=0)

bmdr = QueryInstanceBMDR(X, y, kernel='linear')
select = bmdr.select(label_ind, unlab_ind)
print(select)

spal = QueryInstanceSPAL(X, y, kernel='linear')
select = spal.select(label_ind, unlab_ind)
print(select)

lal = QueryInstanceLAL(X, y, mode='LAL_iterative')
# lal.download_data()
# lal.train_selector_from_file()
select = lal.select(label_ind, unlab_ind)
print(select)
