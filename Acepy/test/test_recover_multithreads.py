# from sklearn import linear_model
# from sklearn.datasets import load_iris
#
# from analyser.experiment_analyser import ExperimentAnalyser
# from data_process.al_split import split
# from experiment_saver.al_experiment import AlExperiment
# from experiment_saver.state import State
# from experiment_saver.state_io import StateIO
# from oracle.oracle import Oracle
# # QBC
# # QBC_ve
# # random
# # uncertainty
# from query_strategy.query_strategy import (QueryInstanceQBC,
#                                            QueryInstanceUncertainty,
#                                            QueryRandom)
# from utils.al_collections import IndexCollection
# from utils.multi_thread import aceThreading
#
# X, y = load_iris(return_X_y=True)
# Train_idx, Test_idx, L_pool, U_pool = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.2, split_count=10)
# ea = ExperimentAnalyser()
# reg = linear_model.LogisticRegression()
# qs = QueryInstanceQBC(X,y,_disagreement='vote_entropy')
# ae = AlExperiment(method_name='QBC_ve')
#
#
# def run_thread(round, train_id, test_id, Lcollection, Ucollection, _saver, _examples, _labels, global_parameters):
#     # initialize object
#     reg.fit(X=_examples[Lcollection.index, :], y=_labels[Lcollection.index])
#     pred = reg.predict(X[test_id, :])
#     accuracy = sum(pred == y[test_id]) / len(test_id)
#     # initialize StateIO module
#     _saver.set_initial_point(accuracy)
#     while len(Ucollection) > 10:
#         select_index = qs.select(Lcollection, Ucollection, reg, n_jobs=1)
#         Ucollection.difference_update(select_index)
#         Lcollection.update(select_index)
#
#         # update model
#         reg.fit(X=_examples[Lcollection.index, :], y=_labels[Lcollection.index])
#         pred = reg.predict(_examples[test_id, :])
#         accuracy = sum(pred == _labels[test_id]) / len(test_id)
#
#         # save intermediate results
#         st = State(select_index=select_index, performance=accuracy)
#         # add user defined information
#         # st.add_element(key='sub_ind', value=sub_ind)
#         _saver.add_state(st)
#         _saver.save()
#
# mt = aceThreading.recover('./multi_thread_state.pkl')
# mt.start_all_threads()
# ae.add_folds(mt._saver)
# ea.add_method(ae)
#
# print(ea)
# ea.simple_plot()