import copy
import numpy as np
from sklearn.datasets import load_iris
from acepy.data_manipulate.al_split import split_features
from acepy.query_strategy.query_features import QueryFeatureAFASMC
from acepy.index import MultiLabelIndexCollection

# from sklearn.preprocessing import MultiLabelBinarizer

X, y = load_iris(return_X_y=True)
# mb = MultiLabelBinarizer()
# mul_y = mb.fit_transform(y)
tr, te, lab, unlab = split_features(feature_matrix=X, test_ratio=0.3, missing_rate=0.5,
                                    split_count=10, all_features=True, saving_path=None)


def test_collec():
    tr0 = tr[0]
    lab0=MultiLabelIndexCollection(lab[0])
    unlabl0=MultiLabelIndexCollection(unlab[0])
    af = QueryFeatureAFASMC(X=X, y=y, train_idx=tr0)
    for i in range(30):
        sel = af.select(observed_entries=lab0, unkonwn_entries=unlabl0)
        lab0.update(sel)
        unlabl0.difference_update(sel)
        print(sel)

def test_1d():
    tr0 = tr[0]
    lab0 = MultiLabelIndexCollection(lab[0])
    unlabl0 = MultiLabelIndexCollection(unlab[0])
    af = QueryFeatureAFASMC(X=X, y=y, train_idx=tr0)
    for i in range(30):
        sel = af.select(observed_entries=lab0, unkonwn_entries=unlabl0)
        lab0.update(sel)
        unlabl0.difference_update(sel)
        print(sel)

def test_mask():
    tr0 = tr[0]
    lab0 = MultiLabelIndexCollection(lab[0])
    unlabl0 = MultiLabelIndexCollection(unlab[0])
    af = QueryFeatureAFASMC(X=X, y=y, train_idx=tr0)

    tr_ob = []
    for entry in lab0:
        assert entry[0] in tr0
        ind_in_train = np.where(tr0 == entry[0])[0][0]
        ind_in_train2 = np.argwhere(tr0 == entry[0])[0][0]
        assert ind_in_train == ind_in_train2
        assert ind_in_train < len(tr0)
        tr_ob.append((ind_in_train, entry[1]))
        # else:
        #     tr_ob.append(entry)
    tr_ob = MultiLabelIndexCollection(tr_ob)

    for i in range(30):
        sel = af.select_by_mask(observed_mask=tr_ob.get_matrix_mask(mat_shape=(len(tr0), X.shape[1]), sparse=False))
        tr_ob.update(sel)
        print(sel)


# a = np.asarray([1,2,3])
# print(np.where(a>1))
# print(np.argwhere(a>1))
test_collec()
# test_1d()
test_mask()
