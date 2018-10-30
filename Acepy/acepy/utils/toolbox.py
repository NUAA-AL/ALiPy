import copy
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target, unique_labels

from acepy.data_manipulate.al_split import split, split_multi_label, split_features
from acepy.experiment.state_io import StateIO
from acepy.oracle.oracle import OracleQueryMultiLabel, Oracle, OracleQueryFeatures
from acepy.experiment.state import State
from acepy.query_strategy.query_strategy import QueryInstanceUncertainty, QueryRandom
from acepy.utils.ace_warnings import *
from acepy.index.index_collections import IndexCollection, MultiLabelIndexCollection, FeatureIndexCollection
from acepy.oracle.knowledge_repository import MatrixRepository, ElementRepository
from acepy.query_strategy.query_type import check_query_type
from acepy.index.multi_label_tools import get_labelmatrix_in_multilabel
from acepy.experiment.stopping_criteria import StoppingCriteria
from acepy.experiment.experiment_analyser import ExperimentAnalyser
from acepy.utils.multi_thread import aceThreading


class ToolBox:
    """Tool box is a tool class which initializes the active learning
    elements according to the setting in order to reduce the error and improve
    the usability.

    In initializing, necessary information to initialize various tool classes
    must be given. You can set the split setting in initializing or generate a
    new split by ToolBox.split.

    Note that, using ToolBox to initialize other tools is optional, you may use
    each modules independently.

    Parameters
    ----------
    y: array-like
        Labels of given data [n_samples, n_labels] or [n_samples]

    X: array-like, optional (default=None)
        data matrix with [n_samples, n_features].

    instance_indexes: array-like, optional (default=None)
        indexes of instances, it should be one-to-one correspondence of
        X, if not provided, it will be generated automatically for each
        x_i started from 0.
        It also can be a list contains names of instances, used for image data_manipulate.
        The split will only depend on the indexes if X is not provided.

    query_type: str, optional (default='AllLabels')
        active learning settings. It will determine how to split data.
        should be one of ['AllLabels', 'Partlabels', 'Features']:

        AllLabels: query all _labels of an selected instance.
            Support scene: binary classification, multi-class classification, multi-label classification, regression

        Partlabels: query part of _labels of an instance.
            Support scene: multi-label classification

        Features: query part of features of an instance.
            Support scene: missing features

    saving_path: str, optional (default='.')
        path to save current settings. if None is provided, then it will not
        save the path

    train_idx: array-like, optional (default=None)
        index of training set, shape like [n_split_count, n_training_indexes]

    test_idx: array-like, optional (default=None)
        index of testing set, shape like [n_split_count, n_testing_indexes]

    label_idx: array-like, optional (default=None)
        index of labeling set, shape like [n_split_count, n_labeling_indexes]

    unlabel_idx: array-like, optional (default=None)
        index of unlabeling set, shape like [n_split_count, n_unlabeling_indexes]


    Attributes
    ----------


    Examples
    ----------

    """

    def __init__(self, y, X=None, instance_indexes=None,
                 query_type='AllLabels', saving_path=None, **kwargs):
        self._index_len = None
        # check and record parameters
        self._y = check_array(y, ensure_2d=False, dtype=None)
        ytype = type_of_target(y)
        if ytype in ['multilabel-indicator', 'multilabel-sequences']:
            self._target_type = 'multilabel'
        else:
            self._target_type = ytype
        self._index_len = len(self._y)
        self._label_space = unique_labels(self._y)
        self._label_num = len(self._label_space)

        self._instance_flag = False
        if X is not None:
            self._instance_flag = True
            self._X = check_array(X, accept_sparse='csr', ensure_2d=True, order='C')
            n_samples = self._X.shape[0]
            if n_samples != self._index_len:
                raise ValueError("Different length of instances and labels found.")
            else:
                self._index_len = n_samples

        if instance_indexes is None:
            self._indexes = [i for i in range(self._index_len)]
        else:
            if len(instance_indexes) != self._index_len:
                raise ValueError("Length of given instance_indexes do not accord the data set.")
            self._indexes = copy.copy(instance_indexes)

        if check_query_type(query_type):
            self.query_type = query_type
            if self.query_type == 'Features' and not self._instance_flag:
                raise Exception("In feature querying, feature matrix must be given.")
        else:
            raise NotImplementedError("Query type %s is not implemented." % type)

        self._split = False
        train_idx = kwargs.pop('train_idx', None)
        test_idx = kwargs.pop('test_idx', None)
        label_idx = kwargs.pop('label_idx', None)
        unlabel_idx = kwargs.pop('unlabel_idx', None)
        if train_idx is not None and test_idx is not None and label_idx is not None and unlabel_idx is not None:
            if not (len(train_idx) == len(test_idx) == len(label_idx) == len(unlabel_idx)):
                raise ValueError("train_idx, test_idx, label_idx, unlabel_idx "
                                 "should have the same split count (length)")
            self._split = True
            self.train_idx = train_idx
            self.test_idx = test_idx
            self.label_idx = label_idx
            self.unlabel_idx = unlabel_idx
            self.split_count = len(train_idx)

        self._saving_path = saving_path
        self._saving_dir = None
        if saving_path is not None:
            if not isinstance(self._saving_path, str):
                raise TypeError("A string is expected, but received: %s" % str(type(self._saving_path)))
            self._saving_path = os.path.abspath(saving_path)
            if os.path.isdir(self._saving_path):
                self._saving_dir = self._saving_path
            else:
                self._saving_dir = os.path.split(self._saving_path)[0]  # if a directory, a dir and None will return.
            self.save()

    def split_AL(self, test_ratio=0.3, initial_label_rate=0.05,
                 split_count=10, all_class=True):
        """split dataset for active learning experiment.
        The labeled set for multi-label setting is fully labeled.

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
        # should support other query types in the future
        self.split_count = split_count
        if self._target_type != 'Features':
            if self._target_type != 'multilabel':
                self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx = split(
                    X=self._X if self._instance_flag else None,
                    y=self._y,
                    query_type=self.query_type, test_ratio=test_ratio,
                    initial_label_rate=initial_label_rate,
                    split_count=split_count,
                    instance_indexes=self._indexes,
                    all_class=all_class,
                    saving_path=self._saving_path)
            else:
                self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx = split_multi_label(
                    y=self._y,
                    label_shape=self._y.shape,
                    test_ratio=test_ratio,
                    initial_label_rate=initial_label_rate,
                    split_count=split_count,
                    all_class=all_class,
                    saving_path=self._saving_path
                )
        else:
            self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx = split_features(
                feature_matrix=self._X,
                test_ratio=test_ratio,
                missing_rate=1 - initial_label_rate,
                split_count=split_count,
                all_features=all_class,
                saving_path=self._saving_path
            )
        self._split = True
        return self.train_idx, self.test_idx, self.label_idx, self.unlabel_idx

    def get_split(self, round=None):
        """Get split of one fold experiment.

        Parameters:
        -----------
        round: int
            The number of fold. 0 <= round < split_count

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
        if not self._split:
            raise Exception("The split setting is unknown, use split_AL() first.")
        if round is not None:
            assert (0 <= round < self.split_count)
            if self.query_type == 'Features':
                return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), FeatureIndexCollection(
                    self.label_idx[round], self._X.shape[1]), FeatureIndexCollection(self.unlabel_idx[round],
                                                                                     self._X.shape[1])
            else:
                if self._target_type == 'multilabel':
                    return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), MultiLabelIndexCollection(
                        self.label_idx[round], self._label_num), MultiLabelIndexCollection(self.unlabel_idx[round],
                                                                                           self._label_num)
                else:
                    return copy.copy(self.train_idx[round]), copy.copy(self.test_idx[round]), IndexCollection(
                        self.label_idx[round]), IndexCollection(self.unlabel_idx[round])
        else:
            return copy.deepcopy(self.train_idx), copy.deepcopy(self.test_idx), \
                   copy.deepcopy(self.label_idx), copy.deepcopy(self.unlabel_idx)

    def get_clean_oracle(self):
        """Get a clean oracle."""
        if self.query_type == 'Features':
            return OracleQueryFeatures(feature_mat=self._X)
        elif self.query_type == 'AllLabels':
            if self._target_type == 'multilabel':
                return OracleQueryMultiLabel(self._y)
            else:
                return Oracle(self._y)

    def get_stateio(self, round, saving_path=None, check_flag=True, verbose=True, print_interval=1):
        """Get a stateio object for experiment saving.

        Parameters:
        -----------
        round: int
            The number of fold. 0 <= round < split_count

        saving_path: str, optional (default='.')
            Path to save the intermediate files. If None is given, it will
            not save the intermediate result.

        check_flag: bool, optional (default=True)
            Whether to check the validity of states.

        verbose: bool, optional (default=True)
            Whether to print query information during the AL process.

        print_interval: int optional (default=1)
            How many queries will trigger a print when verbose is True.

        Returns
        -------
        stateio: StateIO
            The stateio obejct initialized with the specific round.
        """
        assert (0 <= round < self.split_count)
        train_id, test_id, Lcollection, Ucollection = self.get_split(round)
        return StateIO(round, train_id, test_id, Lcollection, Ucollection,
                       saving_path=self._saving_dir if saving_path is None else saving_path,
                       check_flag=check_flag, verbose=verbose, print_interval=print_interval)

    def get_repository(self, round, instance_flag=False):
        """Get knowledge repository object.

        Parameters
        ----------
        round: int
            The number of fold. 0 <= round < split_count

        instance_flag: bool, optional (default=False)
            Whether the repository object contains the examples.
            Note that, if this flag is True, the instances must
            be provided when updating the query information.

        Returns
        -------
        repository: BaseRepository
            knowledge repository object initialized with the labeled set.

        """
        assert (0 <= round < self.split_count)
        train_id, test_id, Ucollection, Lcollection = self.get_split(round)
        if self.query_type == 'AllLabels':
            return MatrixRepository(labels=self._y[Lcollection.index],
                                    examples=self._X[Lcollection.index, :] if instance_flag else None,
                                    indexes=Lcollection.index)
        else:
            return ElementRepository(labels=self._y[Lcollection.index],
                                     examples=self._X[Lcollection.index, :] if instance_flag else None,
                                     indexes=Lcollection.index)

    def get_query_strategy(self, strategy_name="random"):
        """Return the query strategy object.

        Parameters
        ----------
        strategy_name: str, optional (default='random')

        Returns
        -------
        query_strategy: BaseQueryStrategy
            the query_strategy object

        """
        if self.query_type != "AllLabels":
            raise NotImplemented("Query strategy for other query types is not implemented yet.")
        pass

    def default_model(self):
        # return SVC(probability=True, class_weight='balanced')
        return LogisticRegression()

    def stopping_criterion(self, stopping_criteria=None, value=None):
        """Return example stopping criterion.

        Parameters
        ----------
        stopping_criteria: str, optional (default=None)
            stopping criteria, must be one of: [None, 'num_of_queries', 'cost_limit', 'percent_of_unlabel', 'time_limit']

            None: stop when no unlabeled samples available
            'num_of_queries': stop when preset number of quiries is reached
            'cost_limit': stop when cost reaches the limit.
            'percent_of_unlabel': stop when specific percentage of unlabeled data pool is labeled.
            'time_limit': stop when CPU time reaches the limit.

        Returns
        -------
        stop: StoppingCriteria
            The StoppingCriteria object
        """
        return StoppingCriteria(stopping_criteria=stopping_criteria, value=value)

    def experiment_analyser(self, x_axis='num_of_queries'):
        """Return ExperimentAnalyser object

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
        return ExperimentAnalyser(x_axis=x_axis)

    def aceThreading(self, target_function=None, max_thread=None, refresh_interval=1, saving_path='.'):
        """Return the multithreading tool class

        Parameters
        ----------
        target_function: callable, optional (default=None)
            The acceptable active learning main loop.
            the parameters of target_function must be:
            (round, train_id, test_id, Ucollection, Lcollection, saver, examples, labels, global_parameters)
            in which, the global_parameters is a dict which contains the other variables for user-defined function.

        max_thread: int, optional (default=None)
            The max threads for running at the same time. If not provided, it will run all rounds simultaneously.

        refresh_interval: float, optional (default=1.0)
            how many seconds to refresh the current state output, default is 1.0.

        saving_path: str, optional (default='.')
            the path to save the result files.

        Returns
        -------
        ace_threading: aceThreading
            The ace_threading object initialized with the data split.
        """
        if not self._instance_flag:
            raise Exception("instance matrix is necessary for initializing aceThreading object.")
        if not self._split:
            raise Exception("The split information is not found, please split the data or set the split setting first.")
        return aceThreading(examples=self._X, labels=self._y,
                            train_idx=self.train_idx, test_idx=self.test_idx,
                            label_index=self.label_idx,
                            unlabel_index=self.unlabel_idx,
                            refresh_interval=refresh_interval,
                            max_thread=max_thread,
                            saving_path=saving_path,
                            target_func=target_function)

    def save(self):
        """Save the experiment settings to file for auditting or loading for other methods."""
        if self._saving_path is None:
            return
        saving_path = os.path.abspath(self._saving_path)
        if os.path.isdir(saving_path):
            f = open(os.path.join(saving_path, 'al_settings.pkl'), 'wb')
        else:
            f = open(os.path.abspath(saving_path), 'wb')
        pickle.dump(self, f)
        f.close()

    def IndexCollection(self, array=None):
        """Return an IndexCollection object initialized with array"""
        return IndexCollection(array)

    @classmethod
    def load(cls, path):
        """Loading ExperimentSetting object from path.

        Parameters
        ----------
        path: str
            Path to a specific file, not a dir.

        Returns
        -------
        setting: ToolBox
            Object of ExperimentSetting.
        """
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        import pickle
        f = open(os.path.abspath(path), 'rb')
        setting_from_file = pickle.load(f)
        f.close()
        return setting_from_file
