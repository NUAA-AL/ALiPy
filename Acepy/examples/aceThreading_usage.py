from sklearn.datasets import load_iris
from acepy.data_manipulate import split
from acepy.utils import aceThreading

X, y = load_iris(return_X_y=True)
train, test, lab, unlab = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.05,
                                split_count=10)
acethread = aceThreading(examples=X, labels=y,
                         train_idx=train, test_idx=test,
                         label_index=lab, unlabel_index=unlab,
                         max_thread=None)

from sklearn import linear_model
from acepy.experiment import State
from acepy.query_strategy.query_strategy import QueryInstanceQBC


def target_func(round, train_id, test_id, Lcollection, Ucollection, saver, examples, labels, global_parameters):
    # your query strategy
    qs = QueryInstanceQBC(examples, labels, disagreement='vote_entropy')
    # your model
    reg = linear_model.LogisticRegression()
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