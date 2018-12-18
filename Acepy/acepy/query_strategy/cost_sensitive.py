"""

"""
from __future__ import division

import collections
import copy
import warnings
import queue
import random
import matlab
# from decorators import memoized

import numpy as np
from sklearn.svm import SVC
from acepy.utils import interface
from acepy.utils.interface import BaseQueryStrategy 
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
    for i in np.arange(num - 1, 0, -1):
        if (j - costs[i] >= 0) and (dp[i+1][j] == (dp[i][j - costs[i]] + infor_value[i])):
            flag[i] = 1
            j -= costs[i]
    return dp[num][capacity], flag

def select_POSS(infor_value, costs, budget):
    """
    POSS (Pareto Optimization for Subset Selection) method.

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
    population = np.zeros(num)

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
        # compute the fitness of the new solution.
        offspringFit = np.array([0, 0])
        offspringFit[1] = np.sum(offspring * costs)

        if offspringFit[1] == 0 or offspringFit[1] >= 2 * budget:
            offspringFit[0] = np.infty
        else:
            offspringFit[0] = np.sum(offspring * infor_value)
        # use the new solution to update the current population.
        if (fitness[0: popSize, 0] < offspringFit[0] and fitness[0: popSize, 1] <= offspringFit[1]) or (fitness[0: popSize, 0] <= offspringFit[0] and fitness[0: popSize, 1] < offspringFit[1]):
            continue
        else:
            # deleteIndex = fitness[0: popSize, 0] >= offspringFit[0] * fitness[0: popSize, 1] >= offspringFit[1]
            condi_1 = np.where(fitness[0: popSize, 0] < offspringFit[0])
            condi_2 = np.where(fitness[0: popSize, 1] < offspringFit[1])
            nodeleteIndex = [val for val in condi_1[0] if val in condi_2[0]]
            
        # ndelete: record the index of the solutions to be kept.
        population = np.row_stack((population[nodeleteIndex, :], offspring))
        fitness = np.row_stack((fitness[nodeleteIndex, :], offspringFit))
        popSize = len(nodeleteIndex) + 1

    temp = np.where(fitness[:, 1] <= budget)
    max_info_indx = np.argmax(fitness[temp[0], 0])
    max_infovalue = fitness[max_info_indx][0]
    selectedVariables = population[max_info_indx, :]

    return max_infovalue, selectedVariables
    
class HALC(BaseQueryStrategy):
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
    def __init__(self, X=None, y=None, label_index=None, weights=None, label_tree=None, base_model=None, models=None):
    
        # super(costsSensitiveUncertainty, self).__init__(X, y)
        assert(np.shape(X)[0] == np.shape(y)[0])
        assert (isinstance(label_index, collections.Iterable))

        self.X = X
        self.y = y
        self.n_samples, self.n_classes = np.shape(y)
        self.label_index = label_index
        # self.Uncertainty = np.zeros([self.n_samples, self.n_classes])
       
        if weights is None:
            self.weights = np.ones(self.n_classes)
        else:
            assert(np.shape(y)[0] == len(weights))
            self.weights = weights
        if base_model is None:    
            self.base_model = SVC()
        else:
            assert(isinstance(base_model, SVC))
            self.base_model = base_model
        if models is None:
            self.models = self.train_models()
        else:
            self.models = models
        self.Distance = self._cal_Distance()

        # the result of 
        self.Train_Target = np.zeros((self.n_samples, self.n_classes))
        # if node_i is the parent of node_j,then label_tree(i,j)==1
        self.label_tree = label_tree

    def train_models(self):
        """
        Train the models for each class.
        """
        
        train_X = self.X[self.label_index, :]
        train_y = self.y[self.label_index, :]
        models =[]
        for i in np.arange(self.n_classes):       
            i_y = train_y[:, i]
            m = copy.deepcopy(self.base_model)
            m.fit(train_X, i_y)
            models.append(copy.deepcopy(m))   
        return models
    
    def cal_uncertainty(self):
        """
            calculate the uncertainty of instance xi on a label yj.
        Returns: 2d-arrray-like, shape [n_samples, n_classes]
        the uncertainty of instance xi on a label yj.
        """
        Uncertainty = np.zeros([self.n_samples, self.n_classes])
        # unlabel_data = self.X[unlabel_index, :]
        for j in np.arange(self.n_classes):
            model = self.models[j]
            j_target = self.Train_Target[:, j]
            j_label = np.where(j_target != 0)
            j_unlabel = np.where(j_target == 0)
            for i in j_unlabel[0]:
                d_v = model.decision_values(self.X[i][j])
                Uncertainty[i][j] = np.abs(self.weights[j] / d_v)
            Uncertainty[j_label, j] = -np.infty
        
        return Uncertainty

     
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

    def cal_relevance(self, Xi_index, j_class, k=5):
        """
        """
        label_index = np.where(self.Train_Target != 0)
        Xi = self.X[Xi_index, :]
        distance = self.Distance[Xi_index]
        KNN_index = nsmallestarg(distance, k)
        vote = []
        for i in KNN_index:
            if i in label_index[0]:
                g_j = self.y[i, j_class]
            else:
                g_j = self.models[j_class].decision_values(self.X[i, :])
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
    
    def cal_Informativeness(self):
        """

        Returns:
        Info : 2d array-like 
        shape [n_unlabel_samples, n_classes]
        Informativeness of each unlabel samples
        """
        Infor = np.zeros((self.n_samples, self.n_classes))
        Uncertainty = self.cal_uncertainty()
        for j in np.arange(self.n_classes):
            j_target = self.Train_Target[:, j]
            j_label = np.where(j_target != 0)
            j_unlabel = np.where(j_target == 0)
            for i in j_unlabel[0]:
                flag = self.cal_relevance(i, j,k=5)
                if flag == 1:
                    Infor[i][j] = Uncertainty[i][j] * 2
                elif flag == -1:
                    Infor[i][j] = Uncertainty[i][j] + self.cal_Udes(i, j, Uncertainty)
            Infor[j_label][j] = -np.infty

        return Infor
        
    def select(self, label_index, costs, budget):

        Infor = self.cal_Informativeness()
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
        

class QueryRandom(interface.BaseQueryStrategy):
    """Randomly sample a batch of indexes from the unlabel indexes."""

    def select(self, target, cost, budget=40):
        """Select indexes randomly.

        Parameters
        ----------
        unlabel_index: collections.Iterable
            The indexes of unlabeled set.

        batch_size: int, optional (default=1)
            Selection batch size.

        Returns
        -------
        select_pair: 2d array-like, shape[query_instance, query_label]
            The selected indexes which is a subset of unlabel_index.
        """
        n_samples, n_classes = np.shape(target)
        instance_pair = np.array([0, 0])
        costs = 0.
        while True:
            ith_sample = np.random.randint(0, n_samples)   
            jth_class = np.random.choice(np.where(target[ith_sample, :] == 0)[0])
            costs += cost[ith_sample, jth_class]
            if costs >= budget:
                break
            instance_pair = np.row_stack((instance_pair, np.array([ith_sample, jth_class])))

        return instance_pair[1:, ]

                
class QueryUncertainty(BaseQueryStrategy):
    """
    """
    def __init__(self, X=None, y=None):
        super(QueryUncertainty, self).__init__(X, y)
        self.n_samples, self.n_classes = np.shape(y)

    
    def select(self, label_index, costs, models, target, budget):
    
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

    def cal_uncertainty(self, target, models):
        """
        """
        Uncertainty = np.zeros([self.n_samples, self.n_classes])
        # unlabel_data = self.X[unlabel_index, :]
        for j in np.arange(self.n_classes):
            model = models[j]
            j_target = target[:, j]
            j_label = np.where(j_target != 0)
            j_unlabel = np.where(j_target == 0)
            for i in j_unlabel[0]:
                d_v = model.decision_values(self.X[i][j])
                Uncertainty[i][j] = np.abs(1 / d_v)
            Uncertainty[j_label, j] = -np.infty
        
        return Uncertainty

if __name__ == "__main__":
    # a = [1, 1, 1, 1, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15]
    # c = [90, 75, 83, 32, 56, 31, 21, 43, 14, 65, 12, 24, 42, 17, 60]
    # b = 50
    b = 20
    w = [1, 2, 5, 6, 7, 9]
    v = [1, 6, 18, 22, 28, 36]
    # t=select_Knapsack_01(v,w,b)
    k, select=select_Knapsack_01(v,w,b)

    # print(t)

    print(k)
    print(select)