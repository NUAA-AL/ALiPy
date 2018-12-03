import collections

import numpy as np

import acepy.utils.misc


def check_index_multilabel(index):
    """check if the given indexes are legal.

    Parameters
    ----------
    index: list or np.ndarray
        index of the data.
    """
    if not isinstance(index, (list, np.ndarray)):
        index = [index]
    datatype = collections.Counter([type(i) for i in index])
    if len(datatype) != 1:
        raise TypeError("Different types found in the given indexes.")
    if not datatype.popitem()[0] == tuple:
        raise TypeError("Each index should be a tuple.")
    return index


def infer_label_size_multilabel(index_arr, check_arr=True):
    """Infer the label size from a set of index arr.

    raise if all index are example index only.

    Parameters
    ----------
    index_arr: list or np.ndarray
        index array.

    Returns
    -------
    label_size: int
    the inferred label size.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    data_len = np.array([len(i) for i in index_arr])
    if np.any(data_len == 2):
        label_size = np.max([i[1] for i in index_arr if len(i) == 2]) + 1
    elif np.all(data_len == 1):
        raise Exception(
            "Label_size can not be induced from fully labeled set, label_size must be provided.")
    else:
        raise ValueError(
            "All elements in indexes should be a tuple, with length = 1 (example_index, ) "
            "to query all labels or length = 2 (example_index, [label_indexes]) to query specific labels.")
    return label_size


def flattern_multilabel_index(index_arr, label_size=None, check_arr=True):
    """
    Falt the multilabel_index to one-dimensional.

    Parameters
    ----------
    index_arr: list or np.ndarray
        index array.
          
    label_size: int
        the inferred label size.   

    check_arr: bool
        if True,check the index_arr is a legal multilabel_index.
        
    Returns
    -------
    decomposed_data: list
        the decomposed data after falting.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    if label_size is None:
        label_size = infer_label_size_multilabel(index_arr)
    else:
        assert (label_size > 0)
    decomposed_data = []
    for item in index_arr:
        if len(item) == 1:
            for i in range(label_size):
                decomposed_data.append((item[0], i))
        else:
            if isinstance(item[1], collections.Iterable):
                label_ind = [i for i in item[1] if 0 <= i < label_size]
            else:
                assert (0 <= item[1] < label_size)
                label_ind = [item[1]]
            for j in range(len(label_ind)):
                decomposed_data.append((item[0], label_ind[j]))
    return decomposed_data


def integrate_multilabel_index(index_arr, label_size=None, check_arr=True):
    """ Integrated the indexes of multi-label.

    Parameters
    ----------
    index_arr: list or np.ndarray
        multi-label index array.

    label_size: int, optional (default = None)
        the size of label set. If not provided, an inference is attempted.
        raise if the inference is failed.

    check_arr: bool, optional (default = True)
        whether to check the validity of index array.

    Returns
    -------
    array: list
        the integrated array.
    """
    if check_arr:
        index_arr = check_index_multilabel(index_arr)
    if label_size is None:
        label_size = infer_label_size_multilabel(index_arr)
    else:
        assert (label_size > 0)

    integrated_arr = []
    integrated_dict = {}
    for index in index_arr:
        example_ind = index[0]
        if len(index) == 1:
            integrated_dict[example_ind] = set(range(label_size))
        else:
            # length = 2
            if example_ind in integrated_dict.keys():
                integrated_dict[example_ind].update(
                    set(index[1] if isinstance(index[1], collections.Iterable) else [index[1]]))
            else:
                integrated_dict[example_ind] = set(
                    index[1] if isinstance(index[1], collections.Iterable) else [index[1]])

    for item in integrated_dict.items():
        if len(item[1]) == label_size:
            integrated_arr.append((item[0],))
        else:
            # -------------------------------------------------------------------------------------------
            # integrated_arr.append((item[0], tuple(item[0])))
            integrated_arr.append((item[0], tuple(item[1])))

    return integrated_arr


def get_labelmatrix_in_multilabel(index, data_matrix, unknown_element=0):
    """get data matrix by giving index in multi-label setting.

    Note:
    Each index should be a tuple, with the first element representing instance index.
    e.g.
    queried_index = (1, [3,4])  # 1st instance, 3rd,4t _labels
    queried_index = (1, [3])    # 1st instance, 3rd _labels
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all _labels

    Parameters
    ----------
    index: list, np.ndarray or tuple
        if only one index, a tuple is expected.
        Otherwise, it should be a list type with n tuples.

    data_matrix:  array-like
        matrix with [n_samples, n_features] or [n_samples, n_classes].

    unknown_element: object
        value to fill up the unknown part of the matrix_clip.

    Returns
    -------
    Matrix_clip: np.ndarray
        data matrix given index

    index_arr: list
        index of _examples correspond to the each row of Matrix_clip
    """
    # check validity
    index = check_index_multilabel(index)
    data_matrix = acepy.utils.misc.check_matrix(data_matrix)

    ins_bound = data_matrix.shape[0]
    ele_bound = data_matrix.shape[1]

    index_arr = []  # record if a row is already constructed
    current_rows = 0  # record how many rows have been constructed
    label_indexed = None
    for k in index:
        # k is a tuple with 2 elements
        k_len = len(k)
        if k_len != 1 and k_len != 2:
            raise ValueError(
                "A single index should only have 1 element (example_index, ) to query all _labels or"
                "2 elements (example_index, [label_indexes]) to query specific _labels. But found %d in %s" %
                (len(k), str(k)))
        example_ind = k[0]
        assert (example_ind < ins_bound)
        if example_ind in index_arr:
            ind_row = index_arr.index(example_ind)
        else:
            index_arr.append(example_ind)
            ind_row = -1  # new row
            current_rows += 1
        if k_len == 1:  # all _labels
            label_ind = [i for i in range(ele_bound)]
        else:
            if isinstance(k[1], collections.Iterable):
                label_ind = [i for i in k[1] if 0 <= i < ele_bound]
            else:
                assert (0 <= k[1] < ele_bound)
                label_ind = [k[1]]

        # construct mat
        if ind_row == -1:
            tmp = np.zeros((1, ele_bound)) + unknown_element
            tmp[0, label_ind] = data_matrix[example_ind, label_ind]
            if label_indexed is None:
                label_indexed = tmp.copy()
            else:
                label_indexed = np.append(label_indexed, tmp, axis=0)
        else:
            label_indexed[ind_row, label_ind] = data_matrix[example_ind, label_ind]
    return label_indexed, index_arr


def get_Xy_in_multilabel(index, X, y, unknown_element=0):
    """get data matrix by giving index in multi-label setting.

    Note:
    Each index should be a tuple, with the first element representing instance index.
    e.g.
    queried_index = (1, [3,4])  # 1st instance, 3rd,4t _labels
    queried_index = (1, [3])    # 1st instance, 3rd _labels
    queried_index = (1, 3)
    queried_index = (1, (3))
    queried_index = (1, (3,4))
    queried_index = (1, )   # query all _labels

    Parameters
    ----------
    index: list, np.ndarray or tuple
        if only one index, a tuple is expected.
        Otherwise, it should be a list type with n tuples.

    X:  array-like
        array with [n_samples, n_features].

    y:  array-like
        array with [n_samples, n_classes].

    unknown_element: object
        value to fill up the unknown part of the matrix_clip.

    Returns
    -------
    Matrix_clip: np.ndarray
        data matrix given index
    """
    # check validity
    X = acepy.utils.misc.check_matrix(X)
    if not len(X) == len(y):
        raise ValueError("Different length of instances and _labels found.")

    label_matrix, ins_index = get_labelmatrix_in_multilabel(index, y)
    return X[ins_index, :], label_matrix