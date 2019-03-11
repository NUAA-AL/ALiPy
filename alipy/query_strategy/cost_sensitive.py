"""
Implement query strategies for cost-sensitive for hierarchical multi-label setting.

1.Cost-Effective Active Learning for Hierarchical Multi-Label Classification(IJCAI`18)
2.Uncertainty
3.Random
"""
from __future__ import division

import copy
import Queue as queue

import numpy as np
from sklearn.svm import SVC
from ..index import MultiLabelIndexCollection, flattern_multilabel_index, get_Xy_in_multilabel
from .base import BaseMultiLabelQuery
from ..utils.misc import nsmallestarg

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
    for i in range(num):
        for j in range(capacity+1):
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
        in Neural Information Processing Systems, pages 1774-
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
        # compute the fitness of the new solution.
        offspringFit = np.array([0., 0.])
        offspringFit[1] = np.sum(offspring * costs)

        if offspringFit[1] == 0 or offspringFit[1] > budget:
            offspringFit[0] = np.infty
        else:
            offspringFit[0] = np.sum(offspring * infor_value)

        # use the new solution to update the current population.
        # if (fitness[0: popSize, 0] < offspringFit[0] and fitness[0: popSize, 1] <= offspringFit[1]) or (fitness[0: popSize, 0] <= offspringFit[0] and fitness[0: popSize, 1] < offspringFit[1]):
        judge1 = np.array(fitness[0: popSize, 0] < offspringFit[0]) & np.array(fitness[0: popSize, 1] <= offspringFit[1])
        judge2 = np.array(fitness[0: popSize, 0] <= offspringFit[0]) & np.array(fitness[0: popSize, 1] < offspringFit[1])   
        c= judge1 | judge2
        if c.any():
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
    min_info_indx = np.argmin(fitness[temp[0], 0])
    min_infovalue = fitness[min_info_indx][0]
    selectedVariables = population[min_info_indx, :]

    return min_infovalue, selectedVariables

def hierarchical_multilabel_mark(multilabel_index, label_index, label_tree, y_true):
    """"Complete instance-label information according to hierarchy in the label-tree.
    
    Parameters
    ----------
    label_index: {list, np.ndarray, MultiLabelIndexCollection}
        The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
        MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
        the 1st element is the index of instance and the 2nd element is the index of labels.

    multilabel_index: {list, np.ndarray, MultiLabelIndexCollection}
        The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
        MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
        the 1st element is the index of instance and the 2nd element is the index of labels.

    label_tree: np.ndarray
        The hierarchical relationships among data features.
        if node_i is the parent of node_j , then label_tree(i,j)=1

    y_true: 2D array, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_classes]
    
    Returns
    -------
    selected_ins_lab_pair: list
        A list of tuples that contains the indexes of selected instance-label pairs. 
    """
    # try to convert the indexes
    if not isinstance(multilabel_index, MultiLabelIndexCollection):
        try:
            if isinstance(multilabel_index[0], tuple):
                container = MultiLabelIndexCollection(multilabel_index, np.shape(y_true)[1])
            else:
                container = MultiLabelIndexCollection.construct_by_1d_array(multilabel_index, label_mat_shape=np.shape(y_true))
        except:
            raise ValueError(
                "Please pass a 1d array of indexes or MultiLabelIndexCollection (column major, "
                "start from 0) or a list "
                "of tuples with 2 elements, in which, the 1st element is the index of instance "
                "and the 2nd element is the index of label.")
        multilabel_index = copy.deepcopy(container)
    
    if not isinstance(label_index, MultiLabelIndexCollection):
        try:
            if isinstance(label_index[0], tuple):
                container = MultiLabelIndexCollection(label_index, np.shape(y_true)[1])
            else:
                container = MultiLabelIndexCollection.construct_by_1d_array(label_index, label_mat_shape=np.shape(y_true))
        except:
            raise ValueError(
                "Please pass a 1d array of indexes or MultiLabelIndexCollection (column major, "
                "start from 0) or a list "
                "of tuples with 2 elements, in which, the 1st element is the index of instance "
                "and the 2nd element is the index of label.")
        label_index = copy.deepcopy(container)
    
    n_classes = multilabel_index._label_size
    assert(np.shape(label_tree)[0] == n_classes and np.shape(label_tree)[1] == n_classes)

    add_label_index = MultiLabelIndexCollection(label_size=n_classes)
      
    for instance_label_pair in multilabel_index:
        i_instance = instance_label_pair[0]
        j_label = instance_label_pair[1]
        if y_true[instance_label_pair] == 1:
            for descent_label in range(n_classes):
                if label_tree[j_label][descent_label] == 1:
                    if (not (i_instance, descent_label) in label_index):
                        add_label_index.update((i_instance, descent_label))   
        elif y_true[instance_label_pair] == -1:
            for parent_label in range(n_classes):
                if label_tree[parent_label][j_label] == 1:
                    if (not (i_instance, parent_label) in label_index):
                        add_label_index.update((i_instance, parent_label))

    for i in add_label_index:
        if not i in multilabel_index:
            multilabel_index.update(i)
    return multilabel_index
                

class QueryCostSensitiveHALC(BaseMultiLabelQuery):
    """HALC exploit the label hierarchies for cost-effective queries and will selects a 
    batch of instance-label pairs with most information and least cost.
    Select some instance-label pairs based on the Informativeness for Hierarchical Labels
    The definition of  Informativeness for Hierarchical Labels is
            Infor(x,y)=I(y==1)*Uanc + I(y==-1)*Udec + Ux,y;   
            where x is sample,y is label.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_features]

    y: 2D array, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_classes]

    label_tree: 2D array
        The hierarchical relationships among data features.
        if node_i is the parent of node_j , then label_tree(i,j)=1

    weights: np.array, (default=None), shape [1, n_classes] or [n_classes]
        the weights of each class.if not provide,it will all be 1 

    References
    ----------
    [1] Yan Y, Huang S J. Cost-Effective Active Learning for Hierarchical
        Multi-Label Classification[C]//IJCAI. 2018: 2962-2968.
    """
    def __init__(self, X, y, label_tree, weights=None):
    
        super(QueryCostSensitiveHALC, self).__init__(X, y)
        self.n_samples, self.n_classes = np.shape(y)

        if weights is None:
            self.weights = np.ones(self.n_classes)
        else:
            assert(np.shape(y)[0] == len(weights))
            self.weights = weights
        self.label_tree = label_tree
        self.Distance = self._cal_Distance()
    
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
            basemodel = SVC(decision_function_shape='ovr')
        train_traget = label_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
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
        label_mat = label_index.get_matrix_mask((self.n_samples, self.n_classes),sparse=False)
        unlabel_mat = unlabel_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
        for j in np.arange(self.n_classes):
            model = models[j]
            j_label = np.where(label_mat[:, j] == 1)
            j_unlabel = np.where(unlabel_mat[:, j] == 1)
            for i in j_unlabel[0]:
                d_v = model.decision_function([self.X[i]])
                Uncertainty[i][j] = np.abs(self.weights[j] / d_v)
            Uncertainty[j_label, j] = -np.infty
        return Uncertainty
     
    def cal_relevance(self, Xi_index, j_class, label_index, models, k=5):
        """
        """
        label_mat = label_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
        label_samples = np.where(label_mat[:, j_class] == 1)
        distance = self.Distance[Xi_index]
        KNN_index = nsmallestarg(distance, k)
        vote = []
        for i in KNN_index:
            if i in label_samples[0]:
                g_j = self.y[i, j_class]
            else:
                g_j = models[j_class].predict([self.X[i, :]])
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
            if np.any(self.label_tree[temp]):
                for i in self.label_tree[temp]:
                    if self.label_tree[temp, i] == 1:
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
        label_mat = label_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
        unlabel_mat = unlabel_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
        for j in np.arange(self.n_classes):
            j_unlabel = np.where(unlabel_mat[:, j] == 1)[0]
            j_label = np.where(unlabel_mat[:, j] != 1)[0]
            for i in j_unlabel:
                flag = self.cal_relevance(i, j, label_index, models, k=5)
                if flag == 1:
                    Infor[i][j] = Uncertainty[i][j] * 2
                elif flag == -1:
                    Infor[i][j] = Uncertainty[i][j] + self.cal_Udes(i, j, Uncertainty)
            Infor[j_label][j] = -np.infty
        return Infor
        
    def select(self, label_index, unlabel_index, cost=None, oracle=None, budget=40, models=None, base_model=None):
        """ Selects a batch of instance-label pairs with most information and least cost.

        Parameters
        ----------
        label_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        cost: np.array, (default=None), shape [1, n_classes] or [n_classes]
            the cost of querying each class.if not provide,it will all be 1. 

        oracle: Oracle,(default=None)
            Oracle indicate the cost for each label.
            Oracle in active learning whose role is to label the given query.And it can also give the cost of 
            each corresponding label.The Oracle includes the label and cost information at least.
            Oracle(labels=labels, cost=cost)

        budget: int, optional (default=40)
            The budget of the select cost.If cost for eatch labels is 1,will degenerate into the batch_size.

        models: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided,it will build the model based the base_model.
        
        base_model: object, optional(default=None)
            The classification model for eatch label,if the models is not provided.It will build a classifi
            -cation model for the multilabel taks.If not provided, SVM with default parameters implemented
             by sklearn will be used.

        Returns
        -------
        selected_ins_lab_pair: list
            A list of tuples that contains the indexes of selected instance-label pairs.      
        """
        label_index = self._check_multi_label_ind(label_index)
        unlabel_index = self._check_multi_label_ind(unlabel_index)

        if models is None:
            models = self.train_models(label_index, base_model)

        if oracle is None and cost is None:
            raise ValueError('There is no information about the cost of each laebl. \
                            Please input Oracle or cost for the label at least.')
        if oracle:
            _, costs = oracle.query_by_index(range(self.n_classes))
        else:
            costs = cost
        
        Infor = self.cal_Informativeness(label_index, unlabel_index, models)
        instance_pair = np.array([0, 0])
        infor_value=np.array([0])
        corresponding_cost = np.array([0],dtype=int)

        # sort the infor in descent way,in the meanwhile record instance label pair
        for j in np.arange(self.n_classes):
            j_info = Infor[:, j]
            sort_index = np.argsort(j_info)
            sort_index = sort_index[::-1]
            sort_index = sort_index[0: budget]
            useless_index = np.where(Infor[sort_index][j] == -np.infty)
            sort_index = sort_index[: budget - len(useless_index)]
            
            j_instance_pair = np.column_stack((sort_index, np.ones(len(sort_index), dtype=int) * j))
            for k in sort_index:
                infor_value = np.r_[infor_value, Infor[k][j]]
            corresponding_cost = np.r_[corresponding_cost, np.ones(len(sort_index),dtype=int) * costs[j]]
            instance_pair = np.vstack((instance_pair, j_instance_pair))
        
        instance_pair = instance_pair[1:,]
        infor_value = infor_value[1:]
        corresponding_cost = corresponding_cost[1:]

        max_value, select_result = select_Knapsack_01(infor_value, corresponding_cost, budget)
        # max_value, select_result = select_POSS(infor_value, corresponding_cost, budget)

        multilabel_index = [tuple(i) for i in list(instance_pair[np.where(select_result!=0)[0]])]
        # return MultiLabelIndexCollection(multilabel_index, label_size=self.n_classes)
        return multilabel_index
        

class QueryCostSensitiveRandom(BaseMultiLabelQuery):
    """Randomly selects a batch of instance-label pairs.

    Parameters
    ----------
    X: 2D array, optional (default=None)
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_features]

    y: 2D array, optional (default=None)
        Label matrix of the whole dataset. It is a reference which will not use additional memory.
        shape [n_samples, n_classes]
    """
    def __init__(self, X=None, y=None):
        super(QueryCostSensitiveRandom, self).__init__(X, y)

    def select(self, label_index, unlabel_index, oracle=None, cost=None, budget=40):
        """Randomly selects a batch of instance-label pairs under the 
        constraints of meeting the budget conditions.

        Parameters
        ----------
        label_index: ignore
            
        unlabel_index: {list, np.ndarray, MultiLabelIndexCollection}
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        oracle: Oracle,(default=None)
            Oracle indicate the cost for each label.
            Oracle in active learning whose role is to label the given query.And it can also give the cost of 
            each corresponding label.The Oracle includes the label and cost information at least.
            Oracle(labels=labels, cost=cost)

        cost: np.array, (default=None), shape [1, n_classes] or [n_classes]
            The costs of querying each class.if not provide,it will all be 1. 

        budget: int, optional (default=40)
            The budget of the select cost.If cost for eatch labels is 1,will degenerate into the batch_size.

        Returns
        -------
        selected_ins_lab_pair: MultiLabelIndexCollection
            The selected instance label pair.    
        """
        unlabel_index = self._check_multi_label_ind(unlabel_index)
        n_classes = unlabel_index._label_size
        assert(len(cost) == n_classes)   

        if oracle is None and cost is None:
            raise ValueError('There is no information about the cost of each laebl. \
                            Please input Oracle or cost for the label at least.')
        if oracle:
            _, costs = oracle.query_by_index(range(n_classes))
        else:
            costs = cost

        instance_pair = MultiLabelIndexCollection(label_size=n_classes)
        un_ind = copy.deepcopy(unlabel_index)
        current_cost = 0.
        while True:
            rand = np.random.choice(len(un_ind))
            i_j = flattern_multilabel_index(un_ind.index)[rand]
            j_class = i_j[1]
            current_cost += costs[j_class]
            if current_cost > budget:
                break
            instance_pair.update(i_j)
            un_ind.difference_update(i_j)
        # return instance_pair
        return [tuple(i) for i in list(instance_pair)]

                
class QueryCostSensitivePerformance(BaseMultiLabelQuery):
    """Selects the most uncertrainty instance-label pairs under the 
    constraints of meeting the budget conditions.

    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    """
    def __init__(self, X=None, y=None):
        super(QueryCostSensitivePerformance, self).__init__(X, y)
        self.n_samples, self.n_classes = np.shape(y)
        # self.labels = np.unique(np.ravel(self.y))

    def select(self, label_index, unlabel_index, oracle=None, cost=None, budget=40, basemodel=None, models=None):
        """Selects the most uncertrainty instance-label pairs under the 
        constraints of meeting the budget conditions.
        
        Parameters
        ----------
        label_index: MultiLabelIndexCollection
            The indexes of labeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        unlabel_index: MultiLabelIndexCollection
            The indexes of unlabeled samples. It should be a 1d array of indexes (column major, start from 0) or
            MultiLabelIndexCollection or a list of tuples with 2 elements, in which,
            the 1st element is the index of instance and the 2nd element is the index of labels.

        oracle: Oracle,(default=None)
            Oracle indicate the cost for each label.
            Oracle in active learning whose role is to label the given query.And it can also give the cost of 
            each corresponding label.The Oracle includes the label and cost information at least.
            Oracle(labels=labels, cost=cost)

        cost: np.array, (default=None), shape [1, n_classes] or [n_classes]
            the cost of querying each class.if not provide,it will all be 1. 

        budget: int, optional (default=40)
            The budget of the select cost.If cost for eatch labels is 1,will degenerate into the batch_size.

        models: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided,it will build the model based the base_model.
        
        base_model: object, optional(default=None)
            The classification model for eatch label,if the models is not provided.It will build a classifi
            -cation model for the multilabel taks.If not provided, SVM with default parameters implemented
             by sklearn will be used.

        Returns
        -------
        selected_ins_lab_pair: list
            A list of tuples that contains the indexes of selected instance-label pairs. 
        """
        if oracle is None and cost is None:
            raise ValueError('There is no information about the cost of each laebl. \
                            Please input Oracle or cost for the label at least.')
        if oracle:
            _, costs = oracle.query_by_index(range(self.n_classes))
        else:
            costs = cost
        if models is None:
            models = self.train_models(label_index, basemodel)


        unlabel_index = self._check_multi_label_ind(unlabel_index)
        target = unlabel_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
        uncertainty = self.cal_uncertainty(target, models)
        instance_pair = np.array([0, 0])
        infor_value=np.array([0])
        corresponding_cost = np.array([0],dtype=int)

        # sort the infor in descent way,in the meanwhile record instance label pair
        for j in np.arange(self.n_classes):
            j_info = uncertainty[:, j]
            sort_index = np.argsort(j_info)
            sort_index = sort_index[::-1]
            sort_index = sort_index[0: budget]
            useless_index = np.where(uncertainty[sort_index][j] == -np.infty)
            sort_index = sort_index[: budget - len(useless_index)]  
            j_instance_pair = np.column_stack((sort_index, np.ones(len(sort_index), dtype=int) * j))
            for k in sort_index:
                infor_value = np.r_[infor_value, uncertainty[k][j]]
            # infor_value = np.r_[infor_value, uncertainty[sort_index][j]]
            corresponding_cost = np.r_[corresponding_cost, np.ones(len(sort_index),dtype=int) * costs[j]]
            instance_pair = np.vstack((instance_pair, j_instance_pair))
        
        instance_pair = instance_pair[1:,]
        infor_value = infor_value[1:]
        corresponding_cost = corresponding_cost[1:]

        _ , select_result = select_Knapsack_01(infor_value, corresponding_cost, budget)
        multilabel_index = [tuple(i) for i in list(instance_pair[np.where(select_result!=0)[0]])]
        # return MultiLabelIndexCollection(multilabel_index, label_size=self.n_classes)
        return multilabel_index

    def train_models(self, label_index, basemodel):
        """
        Train the models for each class.
        """
        if basemodel is None:
            basemodel = SVC(decision_function_shape='ovr')
        label_index = self._check_multi_label_ind(label_index)
        train_target = label_index.get_matrix_mask((self.n_samples, self.n_classes), sparse=False)
        models =[]
        for j in np.arange(self.n_classes):  
            j_target = train_target[:, j]
            i_samples = np.where(j_target!=0)[0]
            m = copy.deepcopy(basemodel)
            m.fit(self.X[i_samples, :], self.y[i_samples, j])
            models.append(m)   
        return models

    def cal_uncertainty(self, target, models):
        """Calculate the uncertainty.
        target: unlabel_martix
        """
        Uncertainty = np.zeros([self.n_samples, self.n_classes])
        # unlabel_data = self.X[unlabel_index, :]
        for j in np.arange(self.n_classes):
            model = models[j]
            j_target = target[:, j]
            j_label = np.where(j_target != 1)
            j_unlabel = np.where(j_target == 1)
            for i in j_unlabel[0]:
                d_v = model.decision_function([self.X[i]])
                Uncertainty[i][j] = np.abs(1 / d_v)
            Uncertainty[j_label, j] = -np.infty
        return Uncertainty
