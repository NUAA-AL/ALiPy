# from sklearn.datasets import load_iris
# from experiment_saver.state_io import StateIO
#
# from data_process.al_split import split
# from experiment_saver.state import State
#
# X, y = load_iris(return_X_y=True)
# Train_idx, Test_idx, L_pool, U_pool = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.2, _split_count=5)
#
# # main loop
# from query_strategy.query_strategy import QueryInstanceUncertainty
# from oracle.oracle import Oracle
# qs = QueryInstanceUncertainty(X, y, measure='margin')
# oracle = Oracle(y)
# from sklearn import linear_model
# reg = linear_model.LogisticRegression()
#
# # recovery
# _saver = StateIO.load('C:\Code\\acepy\AL_result\experiment_result_file_round_0')
# train_id, test_id, Lcollection, Ucollection = _saver.recovery(5)
# reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
#
# while len(Ucollection)>10:
#     select_index = qs.select(Lcollection, Ucollection,reg,batch_size=2)
#     # accerlate version is available
#     # sub_U = Ucollection.random_sampling()
#     values, costs = oracle.query_by_index(select_index)
#     Ucollection.difference_update(select_index)
#     Lcollection.update(select_index)
#     # db is optional
#     # sup_db.update_query(select_index,values,costs)
#     # reg.fit(X=X[Lcollection.index,:], y=sup_db.get_supervise_info(Lcollection))
#
#     # update model
#     reg.fit(X=X[Lcollection.index, :], y=y[Lcollection.index])
#     pred = reg.predict(X[test_id,:])
#     accuracy = sum(pred == y[test_id])
#
#     # save intermediate results
#     st = State(select_index=select_index, queried_label=values, cost=costs, performance=accuracy)
#     # add user defined information
#     # st.add_element(key='sub_ind', value=sub_ind)
#     _saver.add_state(st)
#     _saver.save()