"""

"""
from __future__ import division

import collections
import copy
import warnings
import queue
import random

import numpy as np
from sklearn.svm import SVC
from acepy.index import MultiLabelIndexCollection
from acepy.query_strategy.base import BaseMultiLabelQuery
from acepy.utils.misc import randperm, nlargestarg, nsmallestarg

def select_Knapsack_01(infor_value, costs, capacity):
    """
    Knapsack-0/1 problem using dynamic porgramming.

    Parameters:
    -----------
    infor_value: array-like
        The value corresponding to each item.
    costs: array-like
        The cost corresponding to each item.
    capacity: float
        The capacity of the knapsack.
    Returns:
    -----------
    max_value: float
        The greatest value the knapsack can bring.
    select_index: array-like
        The result of whether to select the item or not,
        1 reprent this item is selected,0 is not.
    """
    assert(len(infor_value) == len(costs))
    num = len(infor_value)
    dp = np.zeros((num + 1, capacity + 1))
    flag = np.zeros(num)
    for i in np.arange(num):
        for j in np.arange(capacity+1):
            if (j - costs[i]) < 0:
                dp[i+1][j] = dp[i][j]
            else:
                dp[i+1][j] = max(dp[i][j], dp[i][j - costs[i]] + infor_value[i])
    j = capacity
    for i in np.arange(num - 1, -1, -1):
        if (j - costs[i] >= 0) and (dp[i+1][j] == (dp[i][j - costs[i]] + infor_value[i])):
            flag[i] = 1
            j -= costs[i]
    return dp[num][capacity], flag

def select_POSS(infor_value, costs, budget):
    """
    POSS (Pareto Optimization for Subset Selection) method.

    infor-value is negative.
    NOTE THAT we assume that function is to be minimized.
    Paremeters:
    ----------
    infor_value: array-like
        The value corresponding to each item.
    costs: array-like
        The cost corresponding to each item.
    Budget: float
        the constraint on the cost of selected variables.
    Returns:
    ----------
    max_value: float
        The greatest infor-value.
    select_index: array-like
        The result of whether to select the item or not,
        1 reprent this item is selected,0 is not.

    References
    ----------
    [1] Chao Qian, Yang Yu, and Zhi-Hua Zhou.
        Subset selection by pareto optimization. In Advances
        in Neural Information Processing Systems, pages 1774–
        1782, 2015.
    """
    assert(len(infor_value) == len(costs))
    num = len(infor_value)
    population = np.zeros((1, num))

    popSize = 1
    fitness = np.zeros((1, 2))
    fitness[0][0] = np.infty

    fitness[0][1] = 0.
    # repeat to improve the population; 
    # the number of iterations is set as 2*e*k^2*n suggested by our theoretical analysis.
    T = 2 * np.e * np.power(budget, 2) * num

    for round in np.arange(T):
        # randomly select a solution from the population and mutate it to generate a new solution.
        offspring = np.abs(population[np.random.randint(0, popSize), :] - np.random.choice([1, 0], size=(num), p=[1/num, 1 - 1/num]))
        # print('offspring:  ', offspring)
        # compute the fitness of the new solution.
        offspringFit = np.array([0., 0.])
        offspringFit[1] = np.sum(offspring * costs)

        if offspringFit[1] == 0 or offspringFit[1] > budget:
            offspringFit[0] = np.infty
        else:
            offspringFit[0] = np.sum(offspring * infor_value)

        # use the new solution to update the current population.
        judge1 = np.array(fitness[0: popSize, 0] < offspringFit[0]) & np.array(fitness[0: popSize, 1] <= offspringFit[1])
        judge2 = np.array(fitness[0: popSize, 0] <= offspringFit[0]) & np.array(fitness[0: popSize, 1] < offspringFit[1])
        # if (fitness[0: popSize, 0] < offspringFit[0] and fitness[0: popSize, 1] <= offspringFit[1]) or (fitness[0: popSize, 0] <= offspringFit[0] and fitness[0: popSize, 1] < offspringFit[1]):
        c= judge1 | judge2
        if c.any():
            # print("no delete")
            continue
        else:
            # deleteIndex = fitness[0: popSize, 0] >= offspringFit[0] * fitness[0: popSize, 1] >= offspringFit[1]
            index = [i for i in range(len(fitness))]
            condi_1 = np.where(fitness[0: popSize, 0] >= offspringFit[0])
            condi_2 = np.where(fitness[0: popSize, 1] >= offspringFit[1])
            deleteIndex = [val for val in condi_1[0] if val in condi_2[0]]         
            nodeleteIndex = [j for j in index if j not in deleteIndex]    
       
        # ndelete: record the index of the solutions to be kept.
        population = np.row_stack((population[nodeleteIndex, :], offspring))
        fitness = np.row_stack((fitness[nodeleteIndex, :], offspringFit))
        popSize = len(nodeleteIndex) + 1

    temp = np.where(fitness[:, 1] <= budget)
    # max_info_indx = np.argmax(fitness[temp[0], 0])
    # max_infovalue = fitness[max_info_indx][0]
    # selectedVariables = population[max_info_indx, :]
    min_info_indx = np.argmin(fitness[temp[0], 0])
    min_infovalue = fitness[min_info_indx][0]
    selectedVariables = population[min_info_indx, :]

    return min_infovalue, selectedVariables
    
class HALC(BaseMultiLabelQuery):
    """
    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_features]

    y: 2D array, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_classes]
    
    costs: 1d array-like, or list 
        The costs value of each class.shape [n_classes]
    


    model: object, optional (default=None)
        Current classification model, should have the 'predict_proba' method for probabilistic output.
        If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

    batch_size: int, optional (default=1)
        Selection batch size.

    costs:np.array, (default=None), shape [1, n_classes] or [n_classes]
        the costs of querying each class.if not provide,it will all be 1 

    weights: np.array, (default=None), shape [1, n_classes] or [n_classes]
        the weights of each class.if not provide,it will all be 1 

    """
    def __init__(self, X=None, y=None, weights=None, label_tree=None):
    
        super(HALC, self).__init__(X, y)
        self.n_samples, self.n_classes = np.shape(y)

        
        if weights is None:
            self.weights = np.ones(self.n_classes)
        else:
            assert(np.shape(y)[0] == len(weights))
            self.weights = weights
        self.Distance = self._cal_Distance()
        # if node_i is the parent of node_j,then label_tree(i,j)==1
        self.label_tree = label_tree
    
    def _cal_Distance(self):
        """
        """
        Distance = np.zeros((self.n_samples, self.n_samples))
        for i in np.arange(self.n_samples):
            xi = [self.X[i, :]]
            rest_data = self.X[i+1: , :]
            n = np.shape(rest_data)[0]
            query_mat = np.repeat(xi, n, axis=0)
            dist_list = np.sqrt(np.sum((rest_data - query_mat)**2,axis=1))
            Distance[i, i+1: ] = dist_list
        Distance = Distance + Distance.T
        return Distance

    def train_models(self, label_index, basemodel):
        """
        Train the models for each class.
        """

        if basemodel is None:
            basemodel = SVC()
        label_index = self._check_multi_label_ind(label_index)
        train_traget = (label_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        models =[]
        for j in np.arange(self.n_classes):  
            j_target = train_traget[:, j]
            i_samples = np.where(j_target!=0)[0]
            m = copy.deepcopy(basemodel)
            m.fit(self.X[i_samples, :], self.y[i_samples, j])
            models.append(m)   
        return models

    def cal_uncertainty(self, label_index, unlabel_index, models):

        Uncertainty = np.zeros((self.n_samples, self.n_classes))
        # unlabel_data = self.X[unlabel_index, :]
        label_mat = (label_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        unlabel_mat = (unlabel_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        for j in np.arange(self.n_classes):
            model = models[j]
            j_label = np.where(label_mat[:, j] == 1)
            j_unlabel = np.where(unlabel_mat[:, j] == 1)
            for i in j_unlabel[0]:
                d_v = model.decision_values(self.X[i][j])
                Uncertainty[i][j] = np.abs(self.weights[j] / d_v)
            Uncertainty[j_label, j] = -np.infty
        return Uncertainty
     
    def cal_relevance(self, Xi_index, j_class, label_index, models, k=5):
        """
        """
        label_mat = (label_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        label_samples = np.where(label_mat[:, j_class] == 1)
        distance = self.Distance[Xi_index]
        KNN_index = nsmallestarg(distance, k)
        vote = []
        for i in KNN_index:
            if i in label_samples[0]:
                g_j = self.y[i, j_class]
            else:
                g_j = models[j_class].decision_values(self.X[i, :])
            vote.append(np.sign(g_j))       
        return np.sign(np.sum(vote)) 
    
    def cal_Udes(self, xi_index, j_class, Uncertainty):
        """
        """
        Udes = 0
        que = queue.Queue()
        que.put(j_class)     
        while not que.empty():
            temp = que.get()
            if np.any(self.label_tree[j_class]):
                for i in self.label_tree[j_class]:
                    if self.label_tree[j_class, i] == 1:
                        que.put(i)
            else:
                Udes += Uncertainty[xi_index][temp]
        return Udes
    
    def cal_Informativeness(self, label_index, unlabel_index, models):
        """

        Returns:
        Info : 2d array-like 
        shape [n_unlabel_samples, n_classes]
        Informativeness of each unlabel samples
        """
        Infor = np.zeros((self.n_samples, self.n_classes))
        Uncertainty = self.cal_uncertainty(label_index, unlabel_index, models)
        label_mat = (label_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        unlabel_mat = (unlabel_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()

        for j in np.arange(self.n_classes):
            j_label = np.where(label_mat[:, j] == 1)
            j_unlabel = np.where(unlabel_mat[:, j] == 1)
            for i in j_unlabel[0]:
                flag = self.cal_relevance(i, j, label_index, models, k=5)
                if flag == 1:
                    Infor[i][j] = Uncertainty[i][j] * 2
                elif flag == -1:
                    Infor[i][j] = Uncertainty[i][j] + self.cal_Udes(i, j, Uncertainty)
            Infor[j_label][j] = -np.infty

        return Infor
        
    def select(self, label_index, unlabel_index, costs, budget, base_model=None, models=None):

        label_index = self._check_multi_label_ind(label_index)
        unlabel_index = self._check_multi_label_ind(unlabel_index)

        if models is None:
            models = self.train_models(label_index, base_model)

        Infor = self.cal_Informativeness(label_index, unlabel_index, models)
        instance_pair = np.array([0, 0])
        infor_value=np.array([0])
        corresponding_cost = np.array([0])

        # sort the infor in descent way,in the meanwhile record instance label pair
        for j in np.arange(self.n_classes):
            j_info = Infor[:, j]
            sort_index = np.argsort(j_info)
            sort_index = sort_index[0: 40]
            sort_index = sort_index[::-1]
            useless_index = np.where(Infor[sort_index][j] == -np.infty)
            sort_index = sort_index[: 40 - len(useless_index[0])]
            
            instance_pair = np.column_stack((sort_index, np.ones(len(sort_index)) * j))
            infor_value = np.append(infor_value, Infor[sort_index][j])
            corresponding_cost = np.append(corresponding_cost, np.ones(len(sort_index)) * costs[j])
        
        instance_pair = instance_pair[1:,]
        infor_value = infor_value[1:]
        corresponding_cost = corresponding_cost[1:]

        max_value, select_result = select_Knapsack_01(infor_value, corresponding_cost, budget)
        # max_value, select_result = select_POSS(infor_value, corresponding_cost, budget)
        return max_value, instance_pair[np.where(select_result!=0)[0]]
        

class MutlilabelQueryRandom(BaseMultiLabelQuery):
    """Randomly sample a batch of indexes from the unlabel indexes."""

    def __init__(self, X, y):
        """
        """
        super(MutlilabelQueryRandom, self).__init__(X, y)
        self.n_samples, self.n_classes = np.shape(y) 

    def select(self, unlabel_index, cost, batch_size=1, budget=40):
        """Select indexes randomly.

        Parameters
        ----------
        unlabel_index: collections.Iterable
            The indexes of unlabeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        select_pair: MultiLabelIndexCollection
            The selected indexes which is a subset of unlabel_index.
        """
        assert(len(cost) == self.n_classes)   
        unlabel_index = self._check_multi_label_ind(unlabel_index)     
        instance_pair = MultiLabelIndexCollection(label_size=self.n_classes)
        costs = 0.
        batch = 0
        while True:
            onedim_index = unlabel_index.onedim_index
            od_ind = np.random.choice(onedim_index)
            i_sample = od_ind // self.n_classes
            j_class = od_ind % self.n_classes
            costs += cost[i_sample, j_class]
            batch += 1
            if costs > budget or batch > batch_size:
                break
            instance_pair.update((i_sample, j_class))
            unlabel_index.difference_update((i_sample, j_class))

        return instance_pair

                
class QueryUncertainty(BaseMultiLabelQuery):
    """
    """
    def __init__(self, X=None, y=None):
        super(QueryUncertainty, self).__init__(X, y)
        self.n_samples, self.n_classes = np.shape(y)

    
    def select(self, label_index, unlabel_index, batch_size=1, budget=40, basemodel=None, models=None):
        
        if models is None:
            models = self.train_models(label_index, basemodel)

        unlabel_index = self._check_multi_label_ind(unlabel_index)
        target = (unlabel_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        uncertainty = self.cal_uncertainty(target, models)
        instance_pair = np.array([0, 0])
        infor_value=np.array([0])
        corresponding_cost = np.array([0])

        # sort the infor in descent way,in the meanwhile record instance label pair
        for j in np.arange(self.n_classes):

            j_info = uncertainty[:, j]
            sort_index = np.argsort(j_info)
            sort_index = sort_index[0: 40]
            sort_index = sort_index[::-1]
            useless_index = np.where(uncertainty[sort_index][j] == -np.infty)
            sort_index = sort_index[: 40 - len(useless_index)]
            
            instance_pair = np.column_stack((sort_index, np.ones(len(sort_index)) * j))
            infor_value = np.append(infor_value, uncertainty[sort_index][j])
            corresponding_cost = np.append(corresponding_cost, np.ones(len(sort_index)) * costs[j])
        
        instance_pair = instance_pair[1:,]
        infor_value = infor_value[1:]
        corresponding_cost = corresponding_cost[1:]

        max_value, select_result = select_Knapsack_01(infor_value, corresponding_cost, budget)
        # max_value, select_result = select_POSS(infor_value, corresponding_cost, budget)
        return max_value, instance_pair[np.where(select_result!=0)[0]]

    def train_models(self, label_index, basemodel):
        """
        Train the models for each class.
        """
        if basemodel is None:
            basemodel = SVC()
        label_index = self._check_multi_label_ind(label_index)
        train_target = (label_index.get_matrix_mask((self.n_samples, self.n_classes))).todense()
        models =[]
        for j in np.arange(self.n_classes):  
            j_target = train_target[:, j]
            i_samples = np.where(j_target!=0)[0]
            m = copy.deepcopy(basemodel)
            m.fit(self.X[i_samples, :], self.y[i_samples, j])
            models.append(m)   
        return models

    def cal_uncertainty(self, target, models):
        """
        """
        Uncertainty = np.zeros([self.n_samples, self.n_classes])
        # unlabel_data = self.X[unlabel_index, :]
        for j in np.arange(self.n_classes):
            model = models[j]
            j_target = target[:, j]
            j_label = np.where(j_target != 1)
            j_unlabel = np.where(j_target == 1)
            for i in j_unlabel[0]:
                d_v = model.decision_values(self.X[i][j])
                Uncertainty[i][j] = np.abs(1 / d_v)
            Uncertainty[j_label, j] = -np.infty
        
        return Uncertainty

if __name__ == "__main__":
    b = 20
    w = [1, 2, 5, 6, 7, 9]
    v1 = [1, 6, 18, 22, 28, 36]
    v = [-1, -6, -18, -22, -28, -36]
    costs = [1, 2, 5]
    value = [-3, -6, -8]
    k, select=select_Knapsack_01(v1,w,20)
    print(k)
    print(select)
    
    x, y =select_POSS(v,w,20)

    print(x)
    print(y)