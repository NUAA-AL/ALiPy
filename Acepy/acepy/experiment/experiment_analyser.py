"""
Class to gathering, process and visualize active learning experiment results.
"""
# Authors: Ying-Peng Tang
# License: BSD 3 clause
from __future__ import division

import collections
import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import prettytable as pt
from scipy import interpolate

from acepy.utils.ace_warnings import *
from acepy.utils.interface import BaseAnalyser
from .state_io import StateIO


def ExperimentAnalyser(x_axis='num_of_queries'):
    """Class to gathering, process and visualize active learning experiment results.

    Normally, the results should be a list which contains k elements. Each element represents
    one fold experiment result.
    Legal result object includes:
        - StateIO object.
        - A list contains n performances for n queries.
        - A list contains n tuples with 2 elements, in which, the first
          element is the x_axis (e.g., iteration, accumulative_cost),
          and the second element is the y_axis (e.g., the performance)

    Functions include:
        - Line chart (different x,y,axis, mean±std bars)
        - Paired t-test

    Parameters
    ----------
    x_axis: str, optional (default='num_of_queries')
        The x_axis when analysing the result.
        x_axis should be one of ['num_of_queries', 'cost'],
        if 'cost' is given, your experiment results must contains the
        cost value for each performance value.

    Returns
    -------
    analyser: BaseAnalyser
        The experiment analyser object

    """
    if x_axis not in ['num_of_queries', 'cost']:
        raise ValueError("x_axis should be one of ['num_of_queries', 'cost'].")
    if x_axis == 'num_of_queries':
        return _NumOfQueryAnalyser()
    else:
        return _CostSensitiveAnalyser()


def _type_of_data(result):
    """Judge type of data is given by the user.

    Returns
    -------
    type: int
        0 - StateIO object.
        1 - A list contains n performances for n queries.
        2 - A list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, cost),
            and the second element is the y_axis (e.g., the performance)
    """
    if isinstance(result[0], StateIO):
        return 0
    elif isinstance(result[0], list):
        if isinstance(result[0][0], collections.Iterable):
            if len(result[0][0]) > 1:
                return 2
        return 1
    else:
        raise ValueError("Illegal result data is given.\n"
                         "Legal result object includes:\n"
                         "\t- StateIO object.\n"
                         "\t- A list contains n performances for n queries.\n"
                         "\t- A list contains n tuples with 2 elements, in which, "
                         "the first element is the x_axis (e.g., iteration, cost),"
                         "and the second element is the y_axis (e.g., the performance)")


class StateIOContainer:
    """Class to process StateIO object.

    If a list of StateIO objects is given, the data stored
    in each StateIO object can be extracted with this class.
    """

    def __init__(self, method_name, method_results):
        self.method_name = method_name
        self.__results = list()
        self.add_folds(method_results)

    def add_fold(self, src):
        """
        Add one fold of active learning experiment.

        Parameters
        ----------
        src: object or str
            StateIO object or path to the intermediate results file.
        """
        if isinstance(src, StateIO):
            self.__add_fold_by_object(src)
        elif isinstance(src, str):
            self.__add_fold_from_file(src)
        else:
            raise TypeError('StateIO object or str is expected, but received:%s' % str(type(src)),
                            category=UnexpectedParameterWarning)

    def add_folds(self, folds):
        """Add multiple folds.

        Parameters
        ----------
        folds: list
            The list contains n StateIO objects.
        """
        for item in folds:
            self.add_fold(item)

    def __add_fold_by_object(self, result):
        """
        Add one fold of active learning experiment

        Parameters
        ----------
        result: utils.StateIO
            object stored a complete fold of active learning experiment
        """
        self.__results.append(copy.deepcopy(result))

    def __add_fold_from_file(self, path):
        """
        Add one fold of active learning experiment from file

        Parameters
        ----------
        path: str
            path of result file.
        """
        f = open(os.path.abspath(path), 'rb')
        result = pickle.load(f)
        f.close()
        assert (isinstance(result, StateIO))
        if not result.check_batch_size():
            warnings.warn('Checking validity fails, different batch size is found.',
                          category=ValidityWarning)
        self.__results.append(copy.deepcopy(result))

    def extract_matrix(self, extract_keys='performance'):
        """Extract the data stored in the StateIO obejct.

        Parameters
        ----------
        extract_keys: str or list of str, optional (default='performance')
            Extract what value in the State object.
            The extract_keys should be a subset of the keys of each State object.
            Such as: 'select_index', 'performance', 'queried_label', 'cost', etc.

            Note that, the extracted matrix is associated with the extract_keys.
            If more than 1 key is given, each element in the matrix is a tuple.
            The values in tuple are one-to-one correspondence to the extract_keys.

        Returns
        -------
        extracted_matrix: list
            The extracted matrix.
        """
        extracted_matrix = []
        if isinstance(extract_keys, str):
            for stateio in self:
                stateio_line = []
                if stateio.initial_point is not None:
                    stateio_line.append(stateio.initial_point)
                for state in stateio:
                    if extract_keys not in state.keys():
                        raise ValueError('The extract_keys should be a subset of the keys of each State object.\n'
                                         'But keys in the state are: %s' % str(state.keys()))
                    stateio_line.append(state.get_value(extract_keys))
                extracted_matrix.append(copy.copy(stateio_line))

        elif isinstance(extract_keys, list):
            for stateio in self:
                stateio_line = []
                for state in stateio:
                    state_line = []
                    for key in extract_keys:
                        if key not in state.keys():
                            raise ValueError('The extract_keys should be a subset of the keys of each State object.\n'
                                             'But keys in the state are: %s' % str(state.keys()))
                        state_line.append(state.get_value(key))
                    stateio_line.append(tuple(state_line))
                extracted_matrix.append(copy.copy(stateio_line))

        else:
            raise TypeError("str or list of str is expected, but received: %s" % str(type(extract_keys)))

        return extracted_matrix

    def to_list(self):
        return copy.deepcopy(self.__results)

    def __len__(self):
        return len(self.__results)

    def __getitem__(self, item):
        return self.__results.__getitem__(item)

    def __iter__(self):
        return iter(self.__results)


class _ContentSummary:
    """
    store summary info of a given method experiment result
    """

    def __init__(self, method_results, method_type):
        self.method_type = method_type
        # basic info
        self.mean = 0
        self.std = 0
        self.folds = len(method_results)

        # for stateio object only
        self.batch_flag = False
        self.ip = None
        self.batch_size = 0

        # Only for num of query
        self.effective_length = 0

        # Only for Cost
        self.cost_inall = []

        if self.method_type == 0:   # A list of StateIO object.
            self.stateio_summary(method_results)
        else:
            self.list_summary(method_results)

    def stateio_summary(self, method_results):
        """Calculate summary of a method.

        Parameters
        ----------
        method_results: utils.AlExperiment.AlExperiment
            experiment results of a method.
        """
        # examine the AlExperiment object
        if not np.all([sio.check_batch_size() for sio in method_results]):
            # warnings.warn('Checking validity fails, different batch size is found.',
            #               category=ValidityWarning)
            self.batch_flag = False
        else:
            bs = np.unique([sio.batch_size for sio in method_results])
            if len(bs) == 1:
                self.batch_flag = True
                self.batch_size = bs[0]

        result_len = [len(sio) for sio in method_results]
        # if len(np.unique(result_len))!=1:
        #     warnings.warn('Checking validity fails, different length of folds is found.',
        #                   category=ValidityWarning)
        self.effective_length = np.min(result_len)

        # get matrix
        ex_data = []
        for result in method_results:
            self.ip = result.initial_point
            one_fold_perf = [result[i].get_value('performance') for i in range(self.effective_length)]
            one_fold_cost = [result[i].get_value('cost') if 'cost' in result[i].keys() else 0 for i in
                             range(self.effective_length)]
            self.cost_inall.append(one_fold_cost)
            if self.ip is not None:
                one_fold_perf.insert(0, self.ip)
            ex_data.append(one_fold_perf)
        mean_ex = np.mean(ex_data, axis=1)
        self.mean = np.mean(mean_ex)
        self.std = np.std(mean_ex)

    def list_summary(self, method_results):
        # Only for num of query
        self.effective_length = np.min([len(i) for i in method_results])
        if self.method_type == 1:
            # basic info
            self.mean = np.mean(method_results)
            self.std = np.std(method_results)
        else:
            perf_mat = [[np.sum(tup[1]) for tup in line] for line in method_results]
            cost_mat = [[tup[0] for tup in line] for line in method_results]
            mean_perf_for_each_fold = [np.mean(perf) for perf in perf_mat]
            self.mean = np.mean(mean_perf_for_each_fold)
            self.std = np.std(mean_perf_for_each_fold)
            # Only for Cost
            self.cost_inall = [np.sum(cost_one_fold) for cost_one_fold in cost_mat]


class _NumOfQueryAnalyser(BaseAnalyser):
    """Class to process the data whose x_axis is the number of query.

    The validity checking will depend only on the number of query.
    """

    def __init__(self):
        super(_NumOfQueryAnalyser, self).__init__()

    def add_method(self, method_name, method_results):
        """
        Add results of a method.

        Parameters
        ----------
        method_results: {list, np.ndarray, StateIOContainer}
            experiment results of a method. contains k stateIO object or
            a list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, accumulative_cost),
            and the second element is the y_axis (e.g., the performance)

        method_name: str
            Name of the given method.
        """
        if isinstance(method_results, (list, np.ndarray)):
            self.__add_list_result(method_name, method_results)
        elif isinstance(method_results, StateIOContainer):
            self.__add_stateio_container(method_name, method_results)
        else:
            raise TypeError('method_results must be one of {list, numpy.ndarray, StateIOContainer}.')

    def __add_stateio_container(self, method_name, method_results):
        self._is_all_stateio = True
        self._data_extracted[method_name] = method_results.extract_matrix()
        self._data_summary[method_name] = _ContentSummary(method_results=method_results.to_list(), method_type=0)

    def __add_list_result(self, method_name, method_results):
        """
        Add results of a method.

        Parameters
        ----------
        method_results: {list, np.ndarray}
            experiment results of a method. contains k stateIO object with k-fold experiment results.

        method_name: str
            Name of the given method.
        """
        assert (isinstance(method_results, (list, np.ndarray)))
        # StateIO object
        # The type must be one of [0,1,2], otherwise, it will raise in that function.
        self._is_all_stateio = True
        result_type = _type_of_data(method_results)
        if result_type == 0:
            method_container = StateIOContainer(method_name=method_name, method_results=method_results)
            self._data_extracted[method_name] = method_container.extract_matrix()
            # get method summary
            # The summary, however, can not be inferred from a list of performances.
            # So if any lists of extracted data are given, we assume all the results are legal,
            # and thus we will not do further checking on it.
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        elif result_type == 1:
            self._data_extracted[method_name] = copy.copy(method_results)
            self._is_all_stateio = False
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        else:
            raise ValueError("The element in each list should be a single performance value.")

    def _check_plotting(self):
        """
        check:
        1.NaN, Inf etc.
        2.methods_continuity
        """
        if not self._check_methods_continuity:
            warnings.warn('Settings among all methods are not the same. The difference will be ignored.',
                          category=ValidityWarning)
        for i in self._data_extracted.keys():
            if np.isnan(self._data_extracted[i]).any() != 0:
                raise ValueError('NaN is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isnan(self._data_extracted[i]) == True))))
            if np.isinf(self._data_extracted[i]).any() != 0:
                raise ValueError('Inf is found in methods %s in %s.' % (
                    i, str(np.argwhere(np.isinf(self._data_extracted[i]) == True))))
        return True

    def _check_methods_continuity(self):
        """
        check if all methods have the same batch size, length and folds

        Returns
        -------
        result: bool
            True if the same, False otherwise.
        """
        first_flag = True
        bs = 0
        el = 0
        folds = 0
        ip = None
        for i in self._data_extracted.keys():
            summary = self._data_summary[i]
            if first_flag:
                bs = summary.batch_size
                el = summary.effective_length
                folds = summary.folds
                ip = summary.ip
                first_flag = False
            else:
                if bs != summary.batch_size or el != summary.effective_length or folds != summary.folds or not isinstance(
                        ip, type(summary.ip)):
                    return False
        return True

    def plot_learning_curves(self, x_shift=None, start_point=None, title=None, std_area=False, std_alpha=0.3,
                             saving_path='.'):
        """plotting the performance curves.

        Parameters
        ----------
        x_shift: float, optional (default=None)
            The shift value of x_axis.
            For example, the original x_axis is np.arange(0,100,1), x_shift = 1,
            then the new x_axis will be np.arange(1,101,1)

        start_point: float, optional (default=None)
            The value of start point. This value will added before the first data
            point for all methods. If not provided, an infer is attempted.

        title: str, optioanl (default=None)
            The tile of the figure.

        std_area: bool, optional (default=False)
            Whether to show the std values of the performance after each query.

        std_alpha: float, optional (default=0.3)
            The alpha value of the std shaded area.
            The smaller the value, the lighter the color.

        saving_path: str, optional (default='.')
            The path to save the image.
            Passing None to disable the saving.

        Returns
        -------
        plt: object
            The matplot object.
        """
        assert len(self._data_extracted) > 0
        if self._is_all_stateio:
            self._check_plotting()

        # plotting
        for i in self._data_extracted.keys():
            points = np.mean(self._data_extracted[i], axis=0)
            if std_area:
                std_points = np.std(self._data_extracted[i], axis=0)
            if x_shift is None:
                if not self._is_all_stateio or self._data_summary[i].ip is None:
                    x_shift = 1
                else:
                    x_shift = 0
            if start_point is not None:
                x_shift = 0
                plt.plot(np.arange(len(points)+1) + x_shift, [start_point] + list(points), label=i)
                if std_area:
                    plt.fill_between(np.arange(len(points)) + x_shift + 1, points - std_points, points + std_points,
                                     interpolate=True, alpha=std_alpha)
            else:
                plt.plot(np.arange(len(points)) + x_shift, points, label=i)
                if std_area:
                    plt.fill_between(np.arange(len(points)) + x_shift, points - std_points, points + std_points,
                                     interpolate=True, alpha=std_alpha)

        # axis & title
        plt.legend(fancybox=True, framealpha=0.5)
        plt.xlabel("Number of queries")
        plt.ylabel("Performance")
        if title is not None:
            plt.title(str(title))

        # saving
        if saving_path is not None:
            saving_path = os.path.abspath(saving_path)
            if os.path.isdir(saving_path):
                plt.savefig(os.path.join(saving_path, 'acepy_plotting.jpg'))
            else:
                plt.savefig(saving_path)
        plt.show()
        return plt

    def __repr__(self):
        """summary of current methods."""
        tb = pt.PrettyTable()
        tb.field_names = ['Methods', 'number_of_queries', 'number_of_different_split', 'performance']
        for i in self._data_extracted.keys():
            summary = self._data_summary[i]
            tb.add_row([i, summary.effective_length, summary.folds,
                        "%.3f ± %.2f" % (summary.mean, summary.std)])
        if self._is_all_stateio:
            tb.add_column('batch_size', [
                self._data_summary[i].batch_size if self._data_summary[i].batch_flag else 'Not_same_batch_size' for i
                in self._data_extracted.keys()])
        return '\n' + str(tb)


class _CostSensitiveAnalyser(BaseAnalyser):
    """Class to process the cost sensitive experiment results.

    The validity checking will depend only on the cost.
    """

    def __init__(self):
        super(_CostSensitiveAnalyser, self).__init__()

    def add_method(self, method_name, method_results):
        """
        Add results of a method.

        Parameters
        ----------
        method_results: {list, np.ndarray, StateIOContainer}
            experiment results of a method. contains k stateIO object or
            a list contains n tuples with 2 elements, in which, the first
            element is the x_axis (e.g., iteration, cost),
            and the second element is the y_axis (e.g., the performance)

        method_name: str
            Name of the given method.
        """
        if isinstance(method_results, (list, np.ndarray)):
            self.__add_list_result(method_name, method_results)
        elif isinstance(method_results, StateIOContainer):
            self.__add_stateio_container(method_name, method_results)
        else:
            raise TypeError('method_results must be one of {list, numpy.ndarray, StateIOContainer}.')

    def __add_stateio_container(self, method_name, method_results):
        self._is_all_stateio = True
        self._data_extracted[method_name] = method_results.extract_matrix(extract_keys=['cost', 'performance'])
        self._data_summary[method_name] = _ContentSummary(method_results=method_results.to_list(), method_type=0)

    def __add_list_result(self, method_name, method_results):
        self._is_all_stateio = True
        result_type = _type_of_data(method_results)
        if result_type == 0:
            method_container = StateIOContainer(method_name=method_name, method_results=method_results)
            self._data_extracted[method_name] = method_container.extract_matrix(extract_keys=['cost', 'performance'])
            # get method summary
            # The summary, however, can not be inferred from a list of performances.
            # So if any lists of extracted data are given, we assume all the results are legal,
            # and thus we will not do further checking on it.
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        elif result_type == 2:
            self._data_extracted[method_name] = copy.copy(method_results)
            self._is_all_stateio = False
            self._data_summary[method_name] = _ContentSummary(method_results=method_results, method_type=result_type)
        else:
            raise ValueError("Illegal result data in cost sensitive setting is given.\n"
                             "Legal result object includes:\n"
                             "\t- StateIO object.\n"
                             "\t- A list contains n tuples with 2 elements, in which, "
                             "the first element is the x_axis (e.g., iteration, cost),"
                             "and the second element is the y_axis (e.g., the performance)")

    def _check_and_get_total_cost(self):
        """Check if the total cost is the same for all folds.

        Returns
        -------
        same: bool
            If the total cost for all folds are the same.

        effective_cost: float
            If the total cost are the same, return the total cost.
            Otherwise, return the min value of total cost for all folds.

        method_cost: dict
            The effective cost for all methods.
        """
        # return value initialize
        effective_cost = set()
        method_cost = dict()

        # gathering information
        for method_name in self._data_extracted.keys():
            total_cost_folds = []
            for fold in self._data_extracted[method_name]:
                total_cost_fold = [np.sum(query_info[0]) for query_info in fold]
                total_cost_folds.append(np.sum(total_cost_fold))

            method_unique_cost = np.unique(total_cost_folds)
            effective_cost.update(set(method_unique_cost))
            method_cost[method_name] = method_unique_cost
        # return
        same = True if len(effective_cost) == 1 else False
        return same, min(effective_cost), method_cost

    def plot_learning_curves(self, x_shift=0, start_point=None, interpolate_interval=None,
                             title=None, std_area=False, std_alpha=0.3, saving_path='.'):
        """plotting the performance curves.

        Parameters
        ----------
        x_shift: float, optional (default=0)
            The shift value of x_axis.
            For example, the original x_axis is np.arange(0,100,1), x_shift = 1,
            then the new x_axis will be np.arange(1,101,1)

        start_point: float, optional (default=None)
            The value of start point. This value will added before the first data
            point for all methods. If not provided, an infer is attempted.


        interpolate_interval: float, optional (default=None)
            The interpolate interval in plotting the cost sensitive curves.
            The interpolate is needed because the x_axis is not aligned due to the different cost of labels.
            If not provided, it will use cost_budget/100 as the default interval.

        title: str, optioanl (default=None)
            The tile of the figure.

        std_area: bool, optional (default=False)
            Whether to show the std values of the performance after each query.

        std_alpha: float, optional (default=0.3)
            The alpha value of the std shaded area.
            The smaller the value, the lighter the color.

        saving_path: str, optional (default='.')
            The path to save the image.
            Passing None to disable the saving.

        Returns
        -------
        plt: object
            The matplot object.
        """
        same, effective_cost, method_cost = self._check_and_get_total_cost()
        interplt_interval = interpolate_interval if interpolate_interval is not None else effective_cost/100

        # plotting
        for i in self._data_extracted.keys():
            # get un-aligned row data
            data_mat = self._data_extracted[i]
            x_axis = [[np.sum(tup[0]) for tup in line] for line in data_mat]
            # calc accumulative cost in x_axis
            for fold_num in range(len(x_axis)):
                ori_data = x_axis[fold_num]
                acc_data = [np.sum(ori_data[0:list_ind+1]) for list_ind in range(len(ori_data))]
                x_axis[fold_num] = acc_data

            y_axis = [[tup[1] for tup in line] for line in data_mat]

            if start_point is None:
                # attempt to infer the start point
                if not self._is_all_stateio or self._data_summary[i].ip is None:
                    pass
                else:
                    for fold_num in range(len(y_axis)):
                        x_axis[fold_num].insert(0, 0)
                        y_axis[fold_num].insert(0, self._data_summary[i].ip)
            else:
                # Use the specified start point
                for fold_num in range(len(y_axis)):
                    x_axis[fold_num].insert(0, 0)
                    y_axis[fold_num].insert(0, start_point)

            # interpolate
            intplt_arr = []
            for fold_num in range(len(x_axis)):  # len(x_axis) == len(y_axis)
                intplt_arr.append(
                    interpolate.interp1d(x=x_axis[fold_num], y=y_axis[fold_num], bounds_error=False, fill_value=0.1))

            new_x_axis = np.arange(max([x[0] for x in x_axis]), effective_cost, interplt_interval)
            new_y_axis = []
            for fold_num in range(len(y_axis)):
                new_y_axis.append(intplt_arr[fold_num](new_x_axis))

            # plot data
            points = np.mean(new_y_axis, axis=0)
            if std_area:
                std_points = np.std(new_y_axis, axis=0)
            plt.plot(new_x_axis + x_shift, points, label=i)
            if std_area:
                plt.fill_between(new_x_axis, points - std_points, points + std_points,
                                 interpolate=True, alpha=std_alpha)

        # axis & title
        plt.legend(fancybox=True, framealpha=0.5)
        plt.xlabel("Number of queries")
        plt.ylabel("Performance")
        if title is not None:
            plt.title(str(title))

        # saving
        if saving_path is not None:
            saving_path = os.path.abspath(saving_path)
            if os.path.isdir(saving_path):
                plt.savefig(os.path.join(saving_path, 'acepy_plotting.jpg'))
            else:
                plt.savefig(saving_path)
        plt.show()
        return plt

    def __repr__(self):
        """summary of current methods."""
        same, effective_cost, method_cost = self._check_and_get_total_cost()
        tb = pt.PrettyTable()
        tb.field_names = ['Methods', 'number_of_different_split', 'performance', 'cost_budget']
        for i in self._data_extracted.keys():
            summary = self._data_summary[i]
            tb.add_row([i, summary.folds,
                        "%.3f ± %.2f" % (summary.mean, summary.std),
                        method_cost[i] if len(method_cost[i]) == 1 else 'Not same budget'])
        return '\n' + str(tb)


if __name__ == "__main__":
    a = [1.2, 2, 3]
    b = [1.6, 2.5, 1.1]
    print(ExperimentAnalyser().paired_ttest(a, b))
    print(ExperimentAnalyser().paired_ttest(a, a))
