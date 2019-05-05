from sklearn import linear_model
from sklearn.datasets import load_iris

from alipy.experiment import ExperimentAnalyser, State, StateIO
from alipy.data_manipulate.al_split import split
# QBC
# QBC_ve
# random
# uncertainty
from alipy.query_strategy import (QueryInstanceQBC,
                                           QueryInstanceUncertainty,
                                           QueryRandom)
from alipy.index.index_collections import IndexCollection
from alipy.utils.multi_thread import aceThreading

def recover():

    X, y = load_iris(return_X_y=True)
    ea = ExperimentAnalyser()
    reg = linear_model.LogisticRegression(solver='liblinear')
    qs = QueryInstanceQBC(X,y,disagreement='vote_entropy')


    # Estimator, performanceMeasure,
    def run_thread(round, train_id, test_id, Lcollection, Ucollection, saver, examples, labels, global_parameters):
        # initialize object
        reg.fit(X=examples[Lcollection.index, :], y=labels[Lcollection.index])
        pred = reg.predict(examples[test_id, :])
        accuracy = sum(pred == labels[test_id]) / len(test_id)
        # initialize StateIO module
        saver.set_initial_point(accuracy)
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
            # add user defined information
            # st.add_element(key='sub_ind', value=sub_ind)
            saver.add_state(st)
            saver.save()

    mt = aceThreading.recover('./multi_thread_state.pkl')
    mt.start_all_threads()
    ea.add_method(method_name='QBC', method_results=mt.get_results())

    print(ea)
    ea.plot_learning_curves(show='False')
