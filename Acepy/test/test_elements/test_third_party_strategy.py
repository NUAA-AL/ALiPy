from acepy.query_strategy.sota_strategy import *


class ActiveLearnings_GraphDensitySampler:
    """Diversity promoting sampling method that uses graph density to determine
    most representative points.

    X, y is the whole data set.
    """

    def __init__(self, X, y, seed):
        self.name = 'graph_density'
        self.X = X
        self.flat_X = self.flatten_X()
        # Set gamma for gaussian kernel to be equal to 1/n_features
        self.gamma = 1. / self.X.shape[1]
        self.compute_graph_density()

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    def compute_graph_density(self, n_neighbor=10):
        # kneighbors graph is constructed using k=10
        connect = kneighbors_graph(self.flat_X, n_neighbor, p=1)
        # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
        # another point, make it vice versa
        neighbors = connect.nonzero()
        inds = zip(neighbors[0], neighbors[1])
        # Graph edges are weighted by applying gaussian kernel to manhattan dist.
        # By default, gamma for rbf kernel is equal to 1/n_features but may
        # get better results if gamma is tuned.
        for entry in inds:
            i = entry[0]
            j = entry[1]
            distance = pairwise_distances(self.flat_X[[i]], self.flat_X[[j]], metric='manhattan')
            distance = distance[0, 0]
            weight = np.exp(-distance * self.gamma)
            connect[i, j] = weight
            connect[j, i] = weight
        self.connect = connect
        # Define graph density for an observation to be sum of weights for all
        # edges to the node representing the datapoint.  Normalize sum weights
        # by total number of neighbors.
        self.graph_density = np.zeros(self.X.shape[0])
        for i in np.arange(self.X.shape[0]):
            self.graph_density[i] = connect[i, :].sum() / (connect[i, :] > 0).sum()
        self.starting_density = copy.deepcopy(self.graph_density)

    def select_batch_(self, N, already_selected, **kwargs):
        # If a neighbor has already been sampled, reduce the graph density
        # for its direct neighbors to promote diversity.
        batch = set()
        self.graph_density[already_selected] = min(self.graph_density) - 1
        while len(batch) < N:
            selected = np.argmax(self.graph_density)
            neighbors = (self.connect[selected, :] > 0).nonzero()[1]
            self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
            batch.add(selected)
            self.graph_density[already_selected] = min(self.graph_density) - 1
            self.graph_density[list(batch)] = min(self.graph_density) - 1
        return list(batch)

    def to_dict(self):
        output = {}
        output['connectivity'] = self.connect
        output['graph_density'] = self.starting_density
        return output


from acepy.data_manipulate.al_split import *
from sklearn.datasets import load_iris

# initialize
X, y = load_iris(return_X_y=True)
Train_idx, Test_idx, L, U = split(X=X, y=y, test_ratio=0.3, initial_label_rate=0.1)
train_id = Train_idx[0].tolist()
test_id = Test_idx[0].tolist()
L = L[0]
U = U[0]
L_in_train = [train_id.index(i) for i in L]


def test_graph_same_select():
    ori_qs = ActiveLearnings_GraphDensitySampler(X[train_id], y[train_id], seed=0)
    select_ind1 = ori_qs.select_batch_(N=1, already_selected=L_in_train)
    real_id = train_id[select_ind1[0]]

    # test graph density
    qs = QueryInstanceGraphDensity(X, y, Train_idx[0])
    select_index = qs.select(label_index=L, unlabel_index=U)
    assert real_id == select_index[0]


