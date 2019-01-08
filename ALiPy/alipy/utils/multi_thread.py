import copy
import inspect
import os
import pickle
import threading
import time

import prettytable as pt

from ..experiment import StateIO

class aceThreading:
    """This class implement multi-threading in active learning for multiple 
    random splits experiments.

    It will display the progress of each thead. When all __threads reach the
    end points, it will return k StateIO objects for analysis.

    Once initialized, it can store and recover from any iterations and breakpoints.

    Note that, this class only provides visualization and file IO for __threads, but
    not implement any __threads. You should construct different __threads by your own,
    and then provide them as parameters for visualization.

    Specifically, the parameters of thread function must be:
    (round, train_id, test_id, Ucollection, Lcollection, saver, examples, labels, global_parameters)
    in which, the global_parameters is a dict which contains the other variables for user-defined function.

    Parameters
    ----------
    examples: array-like
        data matrix, shape like [n_samples, n_features].

    labels:: array-like
        labels of examples. shape like [n_samples] or [n_samples, n_classes] if in the multi-label setting.

    train_idx: array-like
        index of training examples. shape like [n_round, n_training_examples].

    test_idx: array-like
        index of training examples. shape like [n_round, n_testing_examples].

    label_index: array-like
        index of initially labeled _examples. shape like [n_round, n_labeled_examples].

    unlabel_index: array-like
        index of unlabeled examples. shape like [n_round, n_unlabeled_examples].

    max_thread: int, optional (default=None)
        The max threads for running at the same time. If not provided, it will run all rounds simultaneously.

    refresh_interval: float, optional (default=1.0)
        how many seconds to refresh the current state output, default is 1.0.

    saving_path: str, optional (default='.')
        the path to save the result files.
    """

    def __init__(self, examples, labels, train_idx, test_idx, label_index, unlabel_index, target_func=None,
                 max_thread=None, refresh_interval=1, saving_path='.'):
        self._examples = examples
        self._labels = labels
        self._train_idx = train_idx
        self._test_idx = test_idx
        self._label_index = label_index
        self._unlabel_index = unlabel_index
        self._refresh_interval = refresh_interval
        self._saving_path = os.path.abspath(saving_path)
        self._recover_arr = None

        # the path to store results of each thread.
        tp_path = os.path.join(self._saving_path, 'AL_result')
        if not os.path.exists(tp_path):
            os.makedirs(tp_path)

        assert (len(train_idx) == len(test_idx) ==
                len(label_index) == len(unlabel_index))
        self._round_num = len(label_index)
        self.__threads = []
        # for monitoring __threads' progress
        self._saver = [
            StateIO(round=i, train_idx=self._train_idx[i], test_idx=self._test_idx[i], init_U=self._unlabel_index[i],
                    init_L=self._label_index[i], saving_path=os.path.join(self._saving_path, 'AL_result'),
                    verbose=False) for i in range(self._round_num)]
        if max_thread is None:
            self.__max_thread = self._round_num
        else:
            assert max_thread > 0
            self.__max_thread = max_thread
        # for controlling the print frequency
        self._start_time = time.clock()
        # for displaying the time elapse
        self._thread_time_elapse = [-1] * self._round_num
        # for recovering the workspace
        self.__alive_thread = [False] * self._round_num

        self._target_func = None
        if target_func is not None:
            self.set_target_function(target_func)

    def get_results(self):
        """Return the k-fold experiment results."""
        return copy.deepcopy(self._saver)

    def set_target_function(self, target_function):
        """set the active learning main loop function for paralleling.

        Parameters
        ----------
        target_function: function
        """
        # check target function validity
        argname = inspect.getfullargspec(target_function)[0]
        for name1 in ['round', 'train_id', 'test_id', 'Ucollection', 'Lcollection', 'saver', 'examples', 'labels',
                      'global_parameters']:
            if name1 not in argname:
                raise NameError(
                    "the parameters of target_func must be (round, train_id, test_id, "
                    "Ucollection, Lcollection, saver, examples, labels, global_parameters)")
        self._target_func = target_function

    def start_all_threads(self, global_parameters=None):
        """Start multi-threading.

        this function will automatically invoke the thread_func function with the parameters:
        (round, train_id, test_id, Ucollection, Lcollection, saver, examples, labels, **global_parameters),
        in which, the global_parameters should be provided by the user for additional variables.

        It is necessary that the parameters of your thread_func accord the appointment, otherwise,
        it will raise an error.

        Parameters
        ----------
        target_func: function object
            the function to parallel, the parameters must accord the appointment.

        global_parameters: dict, optional (default=None)
            the additional variables to implement user-defined query-strategy.
        """
        if self._target_func is None:
            raise Exception("Function for paralleling is not given, use set_target_function() first.")
        self.__init_threads(global_parameters)
        # start thread
        self.__start_all_threads()

    def __init_threads(self, global_parameters=None):
        if global_parameters is not None:
            assert (isinstance(global_parameters, dict))
        self._global_parameters = global_parameters

        # init thread objects
        for i in range(self._round_num):
            t = threading.Thread(target=self._target_func, name=str(i), kwargs={
                'round': i, 'train_id': self._train_idx[i], 'test_id': self._test_idx[i],
                'Ucollection': self._saver[i].get_workspace()[3], 'Lcollection': self._saver[i].get_workspace()[2],
                'saver': self._saver[i], 'examples': self._examples, 'labels': self._labels,
                'global_parameters': global_parameters})
            self.__threads.append(t)

    def __start_all_threads(self):
        if self._recover_arr is None:
            self._recover_arr = [True] * self._round_num
        else:
            assert (len(self._recover_arr) == self._round_num)
        # start thread
        available_thread = self.__max_thread
        for i in range(self._round_num):
            if not self._recover_arr[i]:
                continue
            if available_thread > 0:
                self.__threads[i].start()
                self._thread_time_elapse[i] = time.time()
                self.__alive_thread[i] = True
                available_thread -= 1

                # saving
                self._update_thread_state()
                self.save()
            else:
                # waiting current thread
                while True:
                    if self._if_refresh():
                        print(self)
                        # The active_count seems also include the main thread
                        # print(threading.active_count())
                    if threading.active_count() - 1 < self.__max_thread:
                        available_thread += self.__max_thread - threading.active_count() + 1
                        break
                # run the if code
                self.__threads[i].start()
                self._thread_time_elapse[i] = time.time()
                self.__alive_thread[i] = True
                available_thread -= 1

                # saving
                self._update_thread_state()
                self.save()

        # waiting for other threads.
        for i in range(self._round_num):
            if not self._recover_arr[i]:
                continue
            while self.__threads[i].is_alive():
                if self._if_refresh():
                    print(self)
            self._update_thread_state()
            self.save()
        print(self)

    def __repr__(self):
        tb = pt.PrettyTable()
        tb.field_names = ['round', 'number_of_queries', 'time_elapse', 'performance (mean ± std)']

        for i in range(len(self._saver)):
            if self._thread_time_elapse[i] == -1:
                time_elapse = '0'
            else:
                time_elapse = time.time() - self._thread_time_elapse[i]
                m, s = divmod(time_elapse, 60)
                h, m = divmod(m, 60)
                time_elapse = "%02d:%02d:%02d" % (h, m, s)
            tb.add_row([self._saver[i].round, len(self._saver[i]),
                        time_elapse,
                        "%.3f ± %.2f" % self._saver[i].get_current_performance()])
        return str(tb)

    def _if_refresh(self):
        if time.clock() - self._start_time > self._refresh_interval:
            self._start_time = time.clock()
            return True
        else:
            return False

    def _update_thread_state(self):
        for i in range(len(self.__threads)):
            if self.__threads[i].is_alive():
                self.__alive_thread[i] = True
            else:
                self.__alive_thread[i] = False

    def __getstate__(self):
        pickle_seq = (
            self._examples,
            self._labels,
            self._train_idx,
            self._test_idx,
            self._label_index,
            self._unlabel_index,
            self._refresh_interval,
            self._saving_path,
            self._round_num,
            self.__max_thread,
            self._target_func,
            self._global_parameters,
            self.__alive_thread,
            self._saver
        )
        return pickle_seq

    def __setstate__(self, state):
        self._examples, self._labels, self._train_idx, self._test_idx, \
        self._label_index, self._unlabel_index, self._refresh_interval, \
        self._saving_path, self._round_num, self.__max_thread, \
        self._target_func, self._global_parameters, self.__alive_thread, self._saver = state

    def save(self):
        """
        Save the information about the current state of multi_thread to the _saving_path in pkl form.
        """
        if self._saving_path is None:
            return
        if os.path.isdir(self._saving_path):
            f = open(os.path.join(self._saving_path, 'multi_thread_state.pkl'), 'wb')
        else:
            f = open(self._saving_path, 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def recover(cls, path):
        """
        Recover the multi_thread_state from path.

        Parameters
        ----------
        path: str
            the path to save the result files.
        """
        # load breakpoint
        if not isinstance(path, str):
            raise TypeError("A string is expected, but received: %s" % str(type(path)))
        f = open(os.path.abspath(path), 'rb')
        breakpoint = pickle.load(f)
        f.close()
        if not isinstance(breakpoint, aceThreading):
            raise TypeError("Please enter the correct path to the multi-threading saving file.")

        # recover the workspace
        # init self
        recover_thread = cls(breakpoint._examples, breakpoint._labels, breakpoint._train_idx,
                             breakpoint._test_idx, breakpoint._label_index, breakpoint._unlabel_index,
                             breakpoint._target_func, breakpoint.__max_thread,
                             breakpoint._refresh_interval, breakpoint._saving_path)
        # loading tmp files
        state_path = os.path.join(breakpoint._saving_path, 'AL_result')
        recover_arr = [True] * breakpoint._round_num
        for i in range(breakpoint._round_num):
            file_dir = os.path.join(state_path, breakpoint._saver[i]._saving_file_name)
            if not breakpoint.__alive_thread[i]:
                if os.path.exists(file_dir) and os.path.getsize(file_dir) != 0:
                    # all finished
                    recover_thread._saver[i] = StateIO.load(
                        os.path.join(state_path, breakpoint._saver[i]._saving_file_name))
                    recover_arr[i] = False
                else:
                    # not started
                    pass
            else:
                if os.path.getsize(file_dir) == 0:
                    # not saving, but started, use the initialized stateIO object
                    continue
                # still running
                # load intermediate result file
                recover_thread._saver[i] = StateIO.load(
                    os.path.join(state_path, breakpoint._saver[i]._saving_file_name))
        recover_thread._recover_arr = recover_arr
        return recover_thread
