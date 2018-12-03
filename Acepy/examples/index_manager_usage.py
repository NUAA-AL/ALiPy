from acepy.index import IndexCollection

# ---------IndexCollection
a = [1, 2, 3]
# a_ind = acebox.IndexCollection(a)
# Or create by importing the module
from acepy.index import IndexCollection

a_ind = IndexCollection(a)
# add a single index, warn if there is a repeated element.
a_ind.add(4)
# discard a single index, warn if not exist.
a_ind.discard(4)
# add a batch of indexes.
a_ind.update([4, 5])
# discard a batch of indexes.
a_ind.difference_update([1, 2])
print(a_ind)

# ---------MultiLabelIndexCollection-------------
from acepy.index import MultiLabelIndexCollection
multi_lab_ind1 = MultiLabelIndexCollection([(0, 1), (0, 2), (0, (3, 4)), (1, (0, 1))], label_size=5)
multi_lab_ind1.update((0, 0))
multi_lab_ind1.update([(1, 2), (1, (3, 4))])
multi_lab_ind1.update([(2,)])
multi_lab_ind1.difference_update([(0,)])
print(multi_lab_ind1)

# ---------Multi-label tools------------------

from acepy.index import flattern_multilabel_index

a_ind = [(1,), (2, [1, 2])]
flattern_multilabel_index(a_ind, label_size=3)
print(a_ind)

from acepy.index import integrate_multilabel_index

a_ind = [(1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
integrate_mul_ind = integrate_multilabel_index(a_ind, label_size=3)
print(integrate_mul_ind)

from acepy.index import get_labelmatrix_in_multilabel
data_matrix = [[1, 1], [2, 2]]
a_ind = [(0, 1), (1, 1)]
matrix_clip, index_arr = get_labelmatrix_in_multilabel(a_ind, data_matrix, unknown_element=-1)
print(index_arr)
print(matrix_clip)


from acepy.index import get_Xy_in_multilabel
X = [[1, 1], [2, 2]]
y = [[3, 3], [4, 4]]
a_ind = [(0, 1), (1, 1)]
X_lab, y_lab = get_Xy_in_multilabel(a_ind, X, y, unknown_element=-1)
print(X_lab)
print(y_lab)

# ---------FeatureIndexCollection-------------
from acepy.index import FeatureIndexCollection

fea_ind1 = FeatureIndexCollection([(0, 1), (0, 2), (0, (3, 4)), (1, (0, 1))], feature_size=5)

print(fea_ind1)
fea_ind1.update((0, 0))
print(fea_ind1)
fea_ind1.update([(1, 2), (1, (3, 4))])
print(fea_ind1)
fea_ind1.difference_update([(0, [3, 4, 2])])
print(fea_ind1)
