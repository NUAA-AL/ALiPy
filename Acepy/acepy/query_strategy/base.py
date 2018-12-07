from abc import abstractmethod

from ..utils.interface import BaseQueryStrategy
from ..oracle import Oracle, Oracles

class BaseIndexQuery(BaseQueryStrategy):
    """The base class for the selection method which imposes a constraint on the parameters of select()"""

    @abstractmethod
    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Select instances to query.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.
        """


class BaseNoisyOracleQuery(BaseQueryStrategy):
    def __init__(self, X, y, oracles):
        super(BaseNoisyOracleQuery, self).__init__(X, y)
        if isinstance(oracles, list):
            self._oracles_type = 'list'
            for oracle in oracles:
                assert isinstance(oracle, Oracle)
        elif isinstance(oracles, Oracles):
            self._oracles_type = 'Oracles'
        else:
            raise TypeError("The type of parameter oracles must be a list or acepy.oracle.Oracles object.")
        self._oracles = oracles

    @abstractmethod
    def select(self, label_index, unlabel_index, batch_size=1, **kwargs):
        """Query from oracles. Return the selected instance, cost and label.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        batch_size: int, optional (default=1)
            Selection batch size.
        """