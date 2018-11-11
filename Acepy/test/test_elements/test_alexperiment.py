from sklearn.datasets import make_classification

from acepy.experiment.al_experiment import AlExperiment


X, y = make_classification(n_samples=150, n_features=20, n_informative=2, n_redundant=2, 
    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, 
    hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

al = AlExperiment(X, y)
al.split_AL()
# +++++++++++test user define query strategy+++++++++
# user_query = QueryInstanceUncertainty
# print(callable(user_query))
# if callable(user_query):
#     print("callable")
#     al.set_query_strategy(user_query, strategyname='QueryUn')
# else:
#     al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='entropy')


# ++++++++++++++test query_strategy++++++++++++
# al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='entropy')
al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='least_confident')
# al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='margin')
# al.set_query_strategy(strategy="QueryInstanceUncertainty", measure='distance_to_boundary')

# al.set_query_strategy('QueryRandom')

# al.set_query_strategy('QureyExpectedErrorReduction')

# al.set_query_strategy('QueryInstanceQBC', method='query_by_bagging', disagreement='vote_entropy')
# al.set_query_strategy('QueryInstanceQBC', method='query_by_bagging', disagreement='KL_divergence')

# +++++++++test sota_stratgey++++++++++++
# al.set_query_strategy('QueryInstanceGraphDensity', metric='manhattan')
# al.set_query_strategy('QueryInstanceQUIRE')




# al.set_performance_metric('zero_one_loss')
al.set_performance_metric('roc_auc_score')

# 返回值不对应
# al.set_performance_metric('f1_score')


# al.start_query(multi_thread=False)
al.start_query()
al.get_experiment_result()
