"""

"""
from __future__ import division

import collections
import copy
import warnings
import queue
# from decorators import memoized

import numpy as np
from sklearn.svm import SVC
from acepy.utils import interface
from acepy.utils.interface import BaseQueryStrategy 
from acepy.utils.misc import randperm, nlargestarg, nsmallestarg

def select_Knapsack_01(infor_value, costs, capacity):
    """
    Returns: 
    max_value: float
        result
    select_index:[insta]
        results 1d-array,1 selected,0 not
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
        for j in np.arange(self.n_samples):
            j_info = Infor[:, j]
            sort_index = np.argsort(j_info)
            sort_index = sort_index[0: 40]
            sort_index = sort_index[::-1]
            useless_index = np.where(Infor[sort_index][j] == -np.infty)
            sort_index = sort_index[: 40 - len(useless_index)]
            
            instance_pair = np.row_stack((sort_index, np.ones(len(sort_index)) * j))
            infor_value = np.append(infor_value, Infor[sort_index][j])
            corresponding_cost = np.append(corresponding_cost, np.ones(len(sort_index)) * costs[j])
        
        max_value, select_result = self.select_Knapsack_01(infor_value, corresponding_cost, budget)
        return max_value, instance_pair[np.where(select_result!=0)[0]]

    def select_Knapsack_01(self, infor_value, costs, capacity):
        """
        Returns: 
        max_value: float
            result
        select_index:[insta]
            results 1d-array,1 selected,0 not
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
        
        
    # def select_POSS(self, ins_lab_pair, con, costs=None, budget):
    #     """
    #     """
    #     if costs is None:
    #         costs = np.ones(self.n_classes)
    #     else:
    #         assert(self.n_classes == len(costs))
    #         costs = costs
    





class QueryRandom(interface.BaseQueryStrategy):
    """Randomly sample a batch of indexes from the unlabel indexes."""

    def select(self, unlabel_index, target, cost, batch_size=1, budget=40):
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
        
        query_cost: float
            the total cost of query the label
        """
        n_samples, n_classes = np.shape(target)
        select_pair = np.zeros((batch_size, n_classes))
        query_cost = 0
        perm = randperm(n_samples - 1, batch_size)
        for i in np.arange(batch_size):
            instacne_target = target[perm[i], :]
            unlabel_class = np.sum(instacne_target == 0)
            query_label = randperm(unlabel_class -1, 1)
            select_pair[i][query_label] == 1
            query_cost += cost[query_label]
            if query_cost >= budget:
                select_pair[i][query_label] == 0
                query_cost -= cost[query_label]
                break
        return select_pair, query_cost

                



# class costssSensitiveUncertainty(BaseQueryStrategy):
#     """
    
#     Parameters
#     ----------
#     X: 2D array, optional (default=None)
#         Feature matrix of the whole dataset. It is a reference which will not use additional memory.
#         shape [n_samples, n_features]

#     y: 2D array, optional (default=None)
#         Label matrix of the whole dataset. It is a reference which will not use additional memory.
#         shape [n_samples, n_classes]
    
#     costs: 1d array-like, or list 
#         The costs value of each class.shape [n_classes]
    


#     model: object, optional (default=None)
#         Current classification model, should have the 'predict_proba' method for probabilistic output.
#         If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

#     batch_size: int, optional (default=1)
#         Selection batch size.

#     costs:

#     weights: np.array, (default=None), shape [1, n_classes] or [n_classes]
#         the weights of each class
#     Returns
#     -------
#     selected_idx: list
#         The selected indexes which is a subset of unlabel_index.
#     """
#     def __init__(self, X=None, y=None, costs=None, weights=None, models=None, batch_size=1):
#         super(costsSensitiveUncertainty, self).__init__(X, y)
#         assert(np.shape(X)[0] == np.shape(y)[0])
#         self.n_samples, self.n_classes = np.shape(y)

#         if costs is None:
#             self.costs = np.ones(self.n_classes)
#         else:
#             assert(np.shape(y)[0] == len(costs))
#             self.costs = costs

#         if weights is None:
#             self.weights = np.ones(self.n_classes)
#         else:
#             assert(np.shape(y)[0] == len(weights))
#             self.weights = weights
                
#         if models is None:
#             self.models = self.train_models()


#     def train_models(self, label_index):
#         """
#         Train the models.
#         """
        
#         train_X = self.X[label_index]
#         train_y = self.y[label_index]
#         models =[]
#         for i in np.arange(self.n_classes):       
#             i_y = train_y[:, i]
#             model = SVC(kernel='linear', C=0.4)
#             model.fit(train_X, i_y)
#             models.append(model)   
#         return models

    



#     def select_uncertainty(self, flag=None, predict_value_all1=None, *args, **kwargs):

#         train_X = self.X[label_index]
#         train_y = self.y[label_index]
#         predict_value_all=[]
#         decision_values = []

#         find_ins = np.zeros(self.n_samples, self.n_classes)

#         if flag == 1:
#             for i in np.arange(self.n_classes):
#                 ii_target = target(arange(),i)
#                 l_ins = find(ii_target == 0)
#                 l_ins2=find(ii_target != 0)
#                 predict_label,accuracy,decision_values=svmpredict(ii_target,data,model(i),'-b 0',nargout=3)
#                 decision_values=abs(self. / decision_values)
#                 decision_values[l_ins2]=- np.Inf
#                 predict_value_all=concat([predict_value_all,decision_values])
#                 predict_value_all0=copy(predict_value_all)
#         else:
#             predict_value_all1[find(target != 0)]=- inf
#             predict_value_all0=copy(predict_value_all1)
#             predict_value_all=copy(predict_value_all1)        
#         con=[]
#         ins_lab_pair=[]
#         for i in arange(1,n_class).reshape(-1):
#             con_i=predict_value_all(arange(),i)

#             max_con,max_ind=sort(con_i,'descend',nargout=2)

#             fi=max_con(arange(1,40))

#             fi[find(fi == - inf)]=[]

#             con=concat([[con],[fi]])

#             ins_lab_pair=concat([[ins_lab_pair],[max_ind(arange(1,size(fi,1))),dot(i,ones(size(fi,1),1))]])
#         budget=40
#         sgbest,selectedVariables=select_KnapsackPSO(ins_lab_pair,costs,con,budget,nargout=2)
#         selectedpair=ins_lab_pair(find(selectedVariables),arange())


#     def select_by_prediction_mat(self, unlabel_index, predict, batch_size=1):
        
#         pass



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