# from sklearn.datasets import load_iris
# from analyser.experiment_analyser import ExperimentAnalyser
# from experiment_saver.al_experiment import AlExperiment, ToolBox
# from sklearn.ensemble import RandomForestClassifier
# from utils.knowledge_db import MatrixKnowledgeDB
# from experiment_saver.state import State
# from sklearn.preprocessing import LabelBinarizer
#
# X, y = load_iris(return_X_y=True)
# # y = LabelBinarizer().fit_transform(y=y_arr)
# # print(y)
# es = ToolBox(X=X, y=y)
# ea = ExperimentAnalyser()
# reg = RandomForestClassifier()
#
# oracle = es.get_clean_oracle()
# qs = es.uncertainty_selection()
#
# ae = AlExperiment(method_name='uncertainty')
# for round in range(5):
#     _saver = es.get_saver(round)
#     train_id, test_id, Lcollection, Ucollection= es.get_split(round)
#     db = es.get_knowledge_db(round)
#     reg.fit(X=db.get_examples(), y=db.get_labels())
#
#     while len(Ucollection) > 10:
#         select_index = qs.select(Lcollection, Ucollection, reg)
#         queried_labels, cost = oracle.query_by_index(select_index)
#         Ucollection.difference_update(select_index)
#         Lcollection.update(select_index)
#         db.update_query(_labels=queried_labels, indexes=select_index, cost=cost, _examples=X[select_index, :])
#         # print(db.retrieve_by_indexes(select_index))
#         # print(db.retrieve_by_examples(X[Lcollection.index, :]))
#
#         # update model
#         reg.fit(X=db.get_examples(), y=db.get_labels())
#         pred = reg.predict(X[test_id, :])
#         accuracy = sum(pred == y[test_id]) / len(test_id)
#
#         st = State(select_index=select_index, performance=accuracy)
#         _saver.add_state(st)
#         _saver.save()
#     ae.add_fold(_saver)
#
# ea.add_method(ae)
# ea.simple_plot()
