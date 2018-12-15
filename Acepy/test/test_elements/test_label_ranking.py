"""
Test the label ranking model
"""

import numpy as np
import scipy.io as scio
from acepy.query_strategy.multi_label import _LabelRankingModel
from acepy.utils.misc import randperm

# generate samples
# n_init = 30
# samples_num = 100
# feature_num = 30
# label_num = 12
# train_data = np.random.rand(samples_num, feature_num)
# train_targets = np.random.randint(2, size=(samples_num, label_num))
# train_targets[train_targets == 0] = -1
# init_idx = randperm(samples_num-1, n_init)
# init_data = train_data[init_idx]
# init_labels = train_targets[init_idx]
ld = scio.loadmat('C:\\git\\AUDI\\generate_data.mat')
init_data = ld['init_data']
init_labels = ld['init_labels']
train_data = ld['train_data']
train_targets = ld['train_targets']


lrmodel = _LabelRankingModel(init_data, init_labels)
B, V, AB, AV, Anum, trounds, costs, norm_up, step_size0, num_sub, \
lmbda, average_begin, average_size, n_repeat, max_query = lrmodel.init_model_train()
# lrmodel.fit(train_data, train_targets, B, V, idxPs, idxNs, costs, norm_up, step_size0, num_sub, AB, AV, Anum, trounds, lmbda,
#             average_begin, average_size)
print(AB,AV,Anum)
BV=lrmodel.get_BV(AB, AV, Anum)
print(BV)
pres,labels=lrmodel.predict(BV,train_data,num_sub)
print(pres.shape)
print(pres)
print(labels)
