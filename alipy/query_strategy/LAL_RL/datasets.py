import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle as pkl

class Dataset:
    """The base class for all datasets.
    
    Every dataset class should inherit from Dataset 
    and load the data. Dataset only declaires the attributes.
    
    Attributes:
        train_data: A numpy array with data that can be labelled.
        train_labels: A numpy array with labels of train_data.
        test_data: A numpy array with data that will be used for testing.
        test_labels: A numpy array with labels of test_data.
        n_state_estimation: An integer indicating #datapoints reserved for state representation estimation.
        distances: A numpy array with pairwise Eucledian distances between all train_data.
    """
    
    def __init__(self, n_state_estimation):
        """Inits the Dataset object and initialises the attributes with given or empty values."""
        self.train_data = np.array([[]])
        self.train_labels = np.array([[]])
        self.test_data = np.array([[]])
        self.test_labels = np.array([[]])
        self.n_state_estimation = n_state_estimation
        self.dataset_name = ""
        self.regenerate()
        
    def regenerate(self):
        """The function for generating a dataset with new parameters."""
        pass
        
    def _scale_data(self):
        """Scales train data to 0 mean and unit variance. Test data is scaled with parameters of train data."""
        scaler = preprocessing.StandardScaler().fit(self.train_data)
        self.train_data = scaler.transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)
        
    def _keep_state_data(self):
        """self.n_state_estimation samples in training data are reserved for estimating the state."""
        self.train_data, self.state_data, self.train_labels, self.state_labels = train_test_split(
            self.train_data, self.train_labels, test_size=self.n_state_estimation)
        
    def _compute_distances(self):
        """Computes the pairwise distances between all training datapoints"""
        self.distances = scipy.spatial.distance.pdist(self.train_data, metric='cosine')
        self.distances = scipy.spatial.distance.squareform(self.distances)
        
        
        
class DatasetUCI(Dataset):      
    """Class for loading standard benchmark classification datasets.
    
    UCI dataset. Can be downloaded here: 
    https://archive.ics.uci.edu/ml/index.php
    
    Attributes:
        possible_names: A list indicating the dataset names that can be used.
        subset: An integer indicating what subset of data to use. 0: even, 1: odd, -1: all datapoints. 
        size: An integer indicating the size of training dataset to sample, if -1 use all data.
        path: the path to the folder that contains the datasets
    """
    
    def __init__(self, possible_names, n_state_estimation, subset, size=-1, path=None):
        """Inits a few attributes and the attributes of Dataset object."""
        self.possible_names = possible_names
        self.subset = subset
        self.size = size
        if path == None:
            self.path = "./dataUCI/"
        else:
            self.path = path
        Dataset.__init__(self, n_state_estimation) 
    
    def regenerate(self):
        """Loads the data and split it into train and test."""
        # every time we select one of the possible datasets to sample data from
        self.dataset_name = np.random.choice(self.possible_names)
        # load data
        data = pkl.load( open( self.path+"/"+self.dataset_name+".p", "rb" ) )
        X = data['X']
        y = data['y']
        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)
        dtst_size = np.size(y)
        
        # even datapoints subset
        if self.subset == 0:
            valid_indeces = list(range(0, dtst_size, 2))
        # odd datapoints subset
        elif self.subset == 1:
            valid_indeces = list(range(1, dtst_size, 2))
        # all datapoints
        elif self.subset == -1:
            valid_indeces = list(range(dtst_size))
        else:
            print('Incorrect subset attribute value!')
        
        # try to split data into training and test subsets while insuring that 
        # all classes from test data are present in train data 
        done = False
        while not done:
            # get a part of dataset according to subset (even, odd or all)
            train_test_data = X[valid_indeces,:]
            train_test_labels = y[valid_indeces,:]
            # use a random half/half split for train and test data
            self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
                train_test_data, train_test_labels, train_size=0.5)
            
            self._scale_data()
            self._keep_state_data()
            self._compute_distances()
            
            # keep only a part of data for training
            self.train_data = self.train_data[:self.size,:]
            self.train_labels = self.train_labels[:self.size,:]
            
            # this is needed to insure that some of the classes are missing in train or test data
            done = len(np.unique(self.train_labels)) == len(np.unique(self.test_labels))
            
