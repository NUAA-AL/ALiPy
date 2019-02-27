from sklearn.datasets import load_iris
from alipy.data_manipulate import split
from alipy.utils.multi_thread import aceThreading
# Get the data
X, y = load_iris(return_X_y=True)
# Split the data
train, test, lab, unlab = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.05,
                                split_count=10)
# init the aceThreading
acethread = aceThreading(examples=X, labels=y,
                         train_idx=train, test_idx=test,
                         label_index=lab, unlabel_index=unlab,
                         max_thread=None, refresh_interval=1, saving_path='.')

from sklearn import linear_model
from alipy.experiment import State
from alipy.query_strategy import QueryInstanceQBC

# define the custom function
# Specifically, the parameters of the custom function must be:
# (round, train_id, test_id, Ucollection, Lcollection, saver, examples, labels, global_parameters)
def target_func(round, train_id, test_id, Lcollection, Ucollection, saver, examples, labels, global_parameters):
    # your query strategy
    qs = QueryInstanceQBC(examples, labels, disagreement='vote_entropy')
    # your model
    reg = linear_model.LogisticRegression(solver='lbfgs')
    reg.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
    # stopping criterion
    while len(Ucollection) > 30:
        select_index = qs.select(Lcollection, Ucollection, reg, n_jobs=1)
        Ucollection.difference_update(select_index)
        Lcollection.update(select_index)

        # update model
        reg.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
        pred = reg.predict(examples[test_id, :])
        accuracy = sum(pred == labels[test_id]) / len(test_id)

        # save intermediate results
        st = State(select_index=select_index, performance=accuracy)
        saver.add_state(st)
        saver.save()      
# set the target function
acethread.set_target_function(target_func)
# start the all threads
acethread.start_all_threads(global_parameters=None)
# get the result,return list of stateIO
stateIO_list = acethread.get_results()
# save the state of multi_thread to the saving_path in pkl form
acethread.save()
#  or Recover the multi_thread_state from path.
recover_acethread = aceThreading.recover("./multi_thread_state.pkl")
