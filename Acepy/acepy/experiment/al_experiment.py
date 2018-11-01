"""
Class to encapsulate various tools
and implement the main loop of active learning.

To run the experiment with only one class,
we have to impose some restrictions to make
sure the robustness of the code.
"""
# Authors: GuoXiang-Li
# License: BSD 3 clause

import os
import pickle
import inspect
import copy

from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array, check_X_y

from acepy.data_manipulate.al_split import split
from acepy.experiment.state_io import StateIO
from acepy.experiment.state import State
from acepy.utils.ace_warnings import *
from acepy.oracle.knowledge_repository import MatrixRepository, ElementRepository
from acepy.experiment.stopping_criteria import StoppingCriteria
from acepy.experiment.experiment_analyser import ExperimentAnalyser
from acepy.utils.multi_thread import aceThreading
import acepy.query_strategy.query_strategy
import acepy.query_strategy.sota_strategy
from acepy.index.index_collections import IndexCollection
from acepy.metrics.performance import accuracy_score


class AlExperiment:
    """AlExperiment is a  class to encapsulate various tools
    and implement the main loop of active learning.
    AlExperiment is used when query-type is 'AllLabels'.
    Only support the most commonly used scenario: query label of an instance

    To run the experiment with only one class,
    we have to impose some restrictions to make
    sure the robustness of the code:
    1. Your model object should accord scikit-learn api
    2. If a custom query strategy is given, you should implement
        the BaseQueryStrategy api. Additional parameters should be static.
    3. The data split should be given if you are comparing multiple methods.
        You may also generate new split with split_AL()


    Parameters
    ----------
    X,y : array
        The data matrix

    model: object
        An model object which accord the scikit-learn api

    performance_metric: str, optional (default='accuracy_score')
        The performance metric

    stopping_criteria: str, optional (default=None)
        stopping criteria, must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']

        None: stop when no unlabeled samples available
        'num_of_queries': stop when preset number of quiries is reached
        'cost_limit': stop when cost reaches the limit.
        'percent_of_unlabel': stop when specific percentage of unlabeled data pool is labeled.
        'time_limit': stop when CPU time reaches the limit.

    batch_size: int, optional (default=1)
        batch size of AL

    train_idx: array-like, optional (default=None)
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: array-like, optional (default=None)
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: array-like, optional (default=None)
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: array-like, optional (default=None)
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]
    """

    def __init__(self, X, y, model=LogisticRegression(), performance_metric='accuracy_score',
                 stopping_criteria=None, stopping_value=None, batch_size=1, **kwargs):
        self.__custom_strategy_flag = False
        self._split = False
        self._metrics = False
        self._split_count = 0

        self._X, self._y = check_X_y(X, y, accept_sparse='csc', multi_output=True)
        self._model = model
        # set default performance metric 
        self._performance_metric_name = performance_metric
        self._performance_metric = accuracy_score
        self._experiment_result = []
        # set split in the initial
        train_idx = kwargs.pop('train_idx', None)
        test_idx = kwargs.pop('test_idx', None)
        label_idx = kwargs.pop('label_idx', None)
        unlabel_idx = kwargs.pop('unlabel_idx', None)
        if train_idx is not None and test_idx is not None and label_idx is not None and unlabel_idx is not None:
            if not (len(train_idx) == len(test_idx) == len(label_idx) == len(unlabel_idx)):
                raise ValueError("train_idx, test_idx, label_idx, unlabel_idx "
                                 "should have the same split count (length)")
            self._split = True
            self._train_idx = train_idx
            self._test_idx = test_idx
            self._label_idx = label_idx
            self._unlabel_idx = unlabel_idx
            self._split_count = len(train_idx)

        self._stopping_criterion = StoppingCriteria(stopping_criteria, stopping_value)
        self._batch_size = batch_size

    def set_query_strategy(self, strategy="QueryInstanceUncertainty", **kwargs):
        """

        Parameters
        ----------
        strategy: {str, callable}, optional (default='QueryInstanceUncertainty')
            The query strategy function.
            Giving str to use a pre-defined strategy
            Giving callable to use a user-defined strategy.

        kwargs: dict, optional
            if kwargs is None,the pre-defined strategy will init in 
            The args used in strategy.
            Note that, each parameters should be static.
            The parameters will be fed to the callable object automatically.
        """
        # user-defined strategy
        if callable(strategy):
            self.__custom_strategy_flag = True
            strategyname = kwargs.pop('strategyname', None)
            if strategyname is not None:
                self._query_function_name = strategyname
            else:
                self._query_function_name = 'user-defined strategy'
            if len(kwargs) == 0:
                self.__custom_func_arg = None
                self._query_function = strategy(self._X, self._y)
            else:    
                self.__custom_func_arg = kwargs
                self._query_function = strategy(self._X, self._y, kwargs)
        else:
            # a pre-defined strategy in Acepy
            if strategy not in ['QueryInstanceQBC', 'QueryInstanceUncertainty', 'QueryRandom', 
                            'QureyExpectedErrorReduction', 'QueryInstanceGraphDensity', 'QueryInstanceQUIRE']:
                raise NotImplementedError('Strategy {} is not implemented. Specify a valid '
                                      'method name or privide a callable object.'.format(str(strategy)))
            else:
                self._query_function_name = strategy
                if strategy == 'QueryInstanceQBC':
                    method = kwargs.pop('method', None)
                    disagreement = kwargs.pop('disagreement', None)
                    self._query_function = acepy.query_strategy.query_strategy.QueryInstanceQBC(self._X, self._y, method, disagreement, scenario)
                elif strategy == 'QueryInstanceUncertainty':
                    measure = kwargs.pop('measure', None)
                    self._query_function = acepy.query_strategy.query_strategy.QueryInstanceUncertainty(self._X, self._y, measure)
                elif strategy == 'QueryRandom':
                    self._query_function = acepy.query_strategy.query_strategy.QueryRandom(self._X, self._y, kwargs)
                elif strategy == 'QureyExpectedErrorReduction':
                    self._query_function = acepy.query_strategy.query_strategy.QureyExpectedErrorReduction(self._X, self._y)
                elif strategy == 'QueryInstanceGraphDensity':
                    if self._train_idx is None:
                        raise ValueError('train_idx is None.Please split data firstly.You can call set_data_split or split_AL to split data.')
                    metric = kwargs.pop('metric', None)
                    self._query_function = acepy.query_strategy.sota_strategy.QueryInstanceGraphDensity(self._X, self._y, self._train_idx, metric)
                elif strategy == 'QueryInstanceQUIRE':
                    if self._train_idx is None:
                        raise ValueError('train_idx is None.Please split data firstly.You can call set_data_split or split_AL to split data.')
                    self._query_function = acepy.query_strategy.sota_strategy.QueryInstanceQUIRE(self._X, self._y, self._train_idx, kwargs)
     
    def set_performance_metric(self, performance_metric='accuracy_score'):
        """
            Set the metric for experiment.
        """
        if performance_metric not in ['accuracy_score', 'roc_auc_score', 'get_fps_tps_thresholds', 'hamming_loss', 'one_error', 'coverage_error',
                                        'label_ranking_loss', 'label_ranking_average_precision_score']:
            raise NotImplementedError('Performance {} is not implemented.'.format(str(performance_metric)))
        
        self._performance_metric_name = performance_metric
        self._performance_metric = getattr(acepy.metrics.performance, performance_metric)
        self._metrics = True

    def set_data_split(self, train_idx, test_idx, label_idx, unlabel_idx):
        """set the data split indexes.

        Parameters
        ----------
        train_idx: array-like, optional (default=None)
            index of training set, shape like [n_split_count, n_training_indexes]

        test_idx: array-like, optional (default=None)
            index of testing set, shape like [n_split_count, n_testing_indexes]

        label_idx: array-like, optional (default=None)
            index of labeling set, shape like [n_split_count, n_labeling_indexes]

        unlabel_idx: array-like, optional (default=None)
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

        Returns
        -------

        """
        if not (len(train_idx) == len(test_idx) == len(label_idx) == len(unlabel_idx)):
            raise ValueError("_train_idx, _test_idx, _label_idx, _unlabel_idx "
                             "should have the same split count (length)")
        self._split = True
        self._train_idx = train_idx
        self._test_idx = test_idx
        self._label_idx = label_idx
        self._unlabel_idx = unlabel_idx
        self._split_count = len(train_idx)

    def split_AL(self, test_ratio=0.3, initial_label_rate=0.05,
                 split_count=10, all_class=True):
        """split dataset for active learning experiment.

        Parameters
        ----------
        test_ratio: float, optional (default=0.3)
            ratio of test set

        initial_label_rate: float, optional (default=0.05)
            ratio of initial label set or the existed features (missing rate = 1-initial_label_rate)
            e.g. initial_labelset*(1-test_ratio)*n_samples

        split_count: int, optional (default=10)
            random split data _split_count times

        all_class: bool, optional (default=True)
            whether each split will contain at least one instance for each class.
            If False, a totally random split will be performed.

        Returns
        -------
        train_idx: list
            index of training set, shape like [n_split_count, n_training_indexes]

        test_idx: list
            index of testing set, shape like [n_split_count, n_testing_indexes]

        label_idx: list
            index of labeling set, shape like [n_split_count, n_labeling_indexes]

        unlabel_idx: list
            index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]

        """
        self._split_count = split_count
        self._split = True
        self._train_idx, self._test_idx, self._label_idx, self._unlabel_idx = split(
            X=self._X,
            y=self._y,
            test_ratio=test_ratio,
            initial_label_rate=initial_label_rate,
            split_count=split_count,
            all_class=all_class)
        return self._train_idx, self._test_idx, self._label_idx, self._unlabel_idx

    def start_query(self, multi_thread=True):
        """Start the active learning main loop
        If using implemented query strategy, It will run in multi-thread default

        """
        if not self._split:
            raise Exception("Data split is unknown. Use set_data_split() to set an existed split, "
                            "or use split_AL() to generate new split.")   
        if not self._metrics:
            raise Exception("Performance_Metrics is unknown." 
                    " Use set_performance_metric() to define a performance_metrics.")                      
        if multi_thread:
            ace = aceThreading(self._X, self._y, self._train_idx, self._test_idx, self._label_idx, self._unlabel_idx)
            ace.set_target_function(self.__al_main_loop)
            ace.start_all_threads()
            self._experiment_result = ace.get_results()
        else:
            for round in range(self._split_count):
                saver = StateIO(round, self._train_idx[round], self._test_idx[round], self._label_idx[round], self._unlabel_idx[round])
                self.__al_main_loop(round, self._train_idx[round], self._test_idx[round], self._label_idx[round], self._unlabel_idx[round], saver)
                self._experiment_result.append(copy.deepcopy(saver))
                
    def __al_main_loop(self, round, train_id, test_id, Lcollection, Ucollection,
                       saver, examples=None, labels=None, global_parameters=None):
        """
            The active-learning main loop.
        """
        Lcollection = IndexCollection(Lcollection)
        Ucollection = IndexCollection(Ucollection)
        self._model.fit(X=self._X[Lcollection.index, :], y=self._y[Lcollection.index])
        pred = self._model.predict(self._X[test_id, :])

        # performance calc
        perf_result = self._performance_metric(pred, self._y[test_id])

        saver.set_initial_point(perf_result)
        while not self._stopping_criterion.is_stop():
            if not self.__custom_strategy_flag:
                if 'model' in inspect.getfullargspec(self._query_function.select)[0]:
                    select_ind = self._query_function.select(Lcollection, Ucollection, batch_size=self._batch_size,
                                                             model=self._model)
                else:
                    select_ind = self._query_function.select(Lcollection, Ucollection, batch_size=self._batch_size)
            else:
                select_ind = self._query_function.select(Lcollection, Ucollection, batch_size=self._batch_size,
                                                         **self.__custom_func_arg)
            Lcollection.update(select_ind)
            Ucollection.difference_update(select_ind)
            # update model
            self._model.fit(X=self._X[Lcollection.index, :], y=self._y[Lcollection.index])
            pred = self._model.predict(self._X[test_id, :])

            # performance calc
            perf_result = self._performance_metric(pred, self._y[test_id])

            # save intermediate results
            st = State(select_index=select_ind, performance=perf_result)
            saver.add_state(st)
            saver.save()
            # update stopping_criteria
            self._stopping_criterion.update_information(saver)
        self._stopping_criterion.reset()

    def get_experiment_result(self, title=None):
        """
            Print the experiment result,and draw a line chart.
        """
        if len(self._experiment_result) == 0:
            raise Exception('There is no experiment result.Use start_query() get experiment result firstly.')
        ea = ExperimentAnalyser()
        ea.add_method(self._query_function_name, self._experiment_result)
        print(ea)
        ea.plot_line_chart(title=title)

    def save(self):
        pass

    def recover(self):
        pass