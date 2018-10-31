import numpy as np
from acepy.metrics.performance import accuracy_score
from acepy.query_strategy.query_strategy import QueryInstanceUncertainty
from acepy.query_strategy.lal_model import LALmodel
from acepy.query_strategy.sota_strategy import QueryInstanceQUIRE
from acepy.utils.ace_warnings import RepeatElementWarning
from acepy.utils.al_collections import IndexCollection
from acepy.utils.base import BaseCollection
from acepy.utils.knowledge_repository import ElementRepository
from acepy.utils.multi_thread import aceThreading
from acepy.utils.query_type import check_query_type
from acepy.utils.stopping_criteria import StoppingCriteria
from acepy.utils.tools import check_one_to_one_correspondence


from acepy.oracle.oracle import Oracle

if __name__ == '__main__':
    print(accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2))))
    Oracle()
    