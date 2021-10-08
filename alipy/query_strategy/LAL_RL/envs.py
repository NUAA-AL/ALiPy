import numpy as np
from sklearn.base import clone
import collections
from sklearn.ensemble import RandomForestClassifier

class LalEnv(object):    
    """The base class for LAL environment.

    Following the conventions of OpenAI gym, 
    this class implements the environment 
    which simulates labelling of a given 
    annotated dataset. The classes differ 
    by the way how the reward is computed
    and when the terminal state is reached.
    It implements the environment that simulates 
    labelling of a given annotated dataset. 

    Attributes:
        dataset: An object of class Dataset.
        model: A classifier from sklearn. 
            Should implement fit, predict and predict_proba.
        model_rf: A random forest classifier that was fit
            to the same data as the data used for model.
        quality_method: A function that computes the quality of the prediction. 
            For example, can be metrics.accuracy_score or metrics.f1_score.                
        n_classes: An integer indicating the number of classes in a dataset.
            Typically 2.
        episode_qualities: A list of floats with the errors of classifiers at various steps.
        n_actions: An integer indicating the possible number of actions 
            (the number of remaining unlabelled points).
        indeces_known: A list of indeces of datapoints whose labels can be used for training.
        indeces_unknown: A list of indeces of datapoint whose labels cannot be used for training yet.
    """
    
    def __init__(self, dataset, model, quality_method):
        """Inits environment with attributes: dataset, model, quality function and other attributes."""
        self.dataset = dataset
        self.model = model
        self.quality_method = quality_method
        # Compute the number of classes as a number of unique labels in train dataset
        self.n_classes = np.size(np.unique(self.dataset.train_labels))
        # Initialise a list where quality at each iteration will be written
        self.episode_qualities = []
    
    def for_lal(self):
        """Function that is used to compute features for lal-regr.
        
        Fits RF classifier to the data."""
        known_data = self.dataset.train_data[self.indeces_known,:]
        known_labels = self.dataset.train_labels[self.indeces_known]
        known_labels = np.ravel(known_labels)        
        self.model_rf = RandomForestClassifier(50, oob_score=True, n_jobs=1)
        self.model_rf.fit(known_data, known_labels)
    
    def reset(self, n_start=2):
        """Resets the environment.
        
        1) The dataset is regenerated accoudring to its method regenerate.
        2) n_start datapoints are selected, at least one datapoint from each class is included.
        3) The model is trained on the initial dataset and the corresponding state of the problem is computed.

        Args:
            n_start: An integer indicating the size of annotated set at the beginning.
            
        Returns:
            classifier_state: a numpy.ndarray characterizing the current classifier 
                of size of number of features for the state,
                in this case it is the size of number of datasamples in dataset.state_data 
            next_action_state: a numpy.ndarray 
                of size #features characterising actions (currently, 3) x #unlabelled datapoints 
                where each column corresponds to the vector characterizing each possible action.
        """

        # SAMPLE INITIAL DATAPOINTS
        self.dataset.regenerate()
        self.episode_qualities = []
        # To train an initial classifier we need at least self.n_classes samples
        if n_start < self.n_classes:
            n_start = self.n_classes
        # Sample n_start datapoints
        self.indeces_known = []
        self.indeces_unknown = []
        for i in np.unique(self.dataset.train_labels):
            # First get 1 point from each class
            cl = np.nonzero(self.dataset.train_labels==i)[0]
            # Insure that we select random datapoints
            indeces = np.random.permutation(cl)
            self.indeces_known.append(indeces[0])
            self.indeces_unknown.extend(indeces[1:])        
        self.indeces_known = np.array(self.indeces_known)
        self.indeces_unknown = np.array(self.indeces_unknown)
        # self.indeces_unknown now containts first all points of class1, then all points of class2 etc.
        # So, we permute them
        self.indeces_unknown = np.random.permutation(self.indeces_unknown)
        # Then, sample the rest of the datapoints at random
        if n_start > self.n_classes:
            self.indeces_known = np.concatenate(([self.indeces_known, self.indeces_unknown[0:n_start-self.n_classes]]))
            self.indeces_unknown = self.indeces_unknown[n_start-self.n_classes:] 
            
        # BUILD AN INITIAL MODEL
        # Get the data corresponding to the selected indeces
        known_data = self.dataset.train_data[self.indeces_known,:]
        known_labels = self.dataset.train_labels[self.indeces_known]
        unknown_data = self.dataset.train_data[self.indeces_unknown,:]
        unknown_labels = self.dataset.train_labels[self.indeces_unknown]
        # Train a model using data corresponding to indeces_known
        known_labels = np.ravel(known_labels)
        self.model.fit(known_data, known_labels)
        # Compute the quality score
        test_prediction = self.model.predict(self.dataset.test_data)
        new_score = self.quality_method(self.dataset.test_labels, test_prediction)
        self.episode_qualities.append(new_score)
        # Get the features categorizing the state        
        classifier_state, next_action_state = self._get_state() 
        self.n_actions = np.size(self.indeces_unknown)    
        return classifier_state, next_action_state
        
    def step(self, action):
        """Makes a step in the environment.

        Follow the action, in this environment it means labelling a datapoint 
        at position 'action' in indeces_unknown.
        
        Args:
            action: An interger indication the position of a datapoint to label.
            
        Returns:
            classifier_state: a numpy.ndarray 
                of size #features characterising state = #datasamples in dataset.state_data 
                that characterizes the current classifier
            next_action_state: a numpy.ndarray 
                of size #features characterising actions (currently, 3) x #unlabelled datapoints 
                where each column corresponds to the vector characterizing each possible action.
            reward: A float with the reward after adding a new datapoint.
            done: A boolean indicator if the episode in terminated.
        """
        # Action indicates the position of a datapoint in self.indeces_unknown 
        # that we want to sample in unknown_data
        # The index in train_data should be retrieved 
        selection_absolute = self.indeces_unknown[action]
        # Label a datapoint: add its index to known samples and removes from unknown
        self.indeces_known = np.concatenate(([self.indeces_known, np.array([selection_absolute])]))
        self.indeces_unknown = np.delete(self.indeces_unknown, action)    
        # Train a model with new labeled data
        known_data = self.dataset.train_data[self.indeces_known,:]
        known_labels = self.dataset.train_labels[self.indeces_known]
        known_labels = np.ravel(known_labels)
        self.model.fit(known_data, known_labels)
        # Get a new state 
        classifier_state, next_action_state = self._get_state() 
        # Update the number of available actions
        self.n_actions = np.size(self.indeces_unknown)
        # Compute the quality of the current classifier
        test_prediction = self.model.predict(self.dataset.test_data)
        new_score = self.quality_method(self.dataset.test_labels, test_prediction)
        self.episode_qualities.append(new_score)
        # Compute the reward
        reward = self._compute_reward()
        # Check if this episode terminated
        done = self._compute_is_terminal()          
        return classifier_state, next_action_state, reward, done
      
    def _get_state(self):
        """Private function for computing the state depending on the classifier and next available actions.
        
        This function computes 1) classifier_state that characterises 
        the current state of the classifier and it is computed as 
        a function of predictions on the hold-out dataset 
        2) next_action_state that characterises all possible actions 
        (unlabelled datapoints) that can be taken at the next step.
        
        Returns:
            classifier_state: a numpy.ndarray 
                              of size of number of datapoints in dataset.state_data 
                              characterizing the current classifier and, thus, the 
                              state of the environment
            next_action_state: a numpy.ndarray 
                               of size #features characterising actions (currently, 3) x #unlabelled datapoints 
                               where each column corresponds to the vector characterizing each possible action.
        """
        # COMPUTE CLASSIFIER_STATE
        predictions = self.model.predict_proba(self.dataset.state_data)[:,0]
        predictions = np.array(predictions)
        idx = np.argsort(predictions)
        # the state representation is the *sorted* list of scores 
        classifier_state = predictions[idx]
        
        # COMPUTE ACTION_STATE
        unknown_data = self.dataset.train_data[self.indeces_unknown,:]
        # prediction (score) of classifier on each unlabelled sample
        a1 = self.model.predict_proba(unknown_data)[:,0]
        # average distance to every unlabelled datapoint
        a2 = np.mean(self.dataset.distances[self.indeces_unknown,:][:,self.indeces_unknown],axis=0)
        # average distance to every labelled datapoint
        a3 = np.mean(self.dataset.distances[self.indeces_known,:][:,self.indeces_unknown],axis=0)
        next_action_state = np.concatenate(([a1], [a2], [a3]), axis=0)
        return classifier_state, next_action_state
    
    def _compute_reward(self):
        """Private function to computes the reward.
        
        Default function always returns 0.
        Every sub-class should implement its own reward function.
        
        Returns:
            reward: a float reward
        """
        reward = 0.0
        return reward
    
    def _compute_is_terminal(self):
        """Private function to compute if the episode has reaches the terminal state.
        
        By default episode terminates when all the data was labelled.
        Every sub-class should implement its own episode termination function.
        
        Returns:
            done: A boolean that indicates if the episode is finished.
        """
        # self.n_actions contains a number of unlabelled datapoints that is left
        if self.n_actions==1:
            # print('We ran out of samples!')
            done = True
        else:
            done = False
        return done
    
    
class LalEnvIncrementalReduction(LalEnv):
    """The LAL environment class with reward that is incremental error reduction.

    This class inherits from LalEnv. 
    The reward is the difference between 
    the test errors at the consequetive 
    iterations. The terminal state is reached 
    when n_horizon samples are labelled.
    
    Attributes:            
        n_horizon: An integer indicating how many steps can be made in an episode.
    """
    
    def __init__(self, dataset, model, quality_method, n_horizon = 10):
        """Inits environment with its normal attributes + n_horizon (the length of the episode)."""
        LalEnv.__init__(self, dataset, model, quality_method)
        self.n_horizon = n_horizon
        
    def _compute_reward(self):
        """Computes the reward.
        
        Computes the reward that is the difference 
        between the previous model score and new model score.
        
        Returns:
            reward: A float reward.
        """
        last_score = self.episode_qualities[-2]
        new_score = self.episode_qualities[-1]
        reward = new_score - last_score
        return reward
    
    def _compute_is_terminal(self):
        """Computes if the episode has reaches the terminal state.
        
        The end of the episode is reached when number 
        of labelled points reaches predifined horizon.
        
        Returns:
            done: A boolean that indicates if the episode is finished.
        """
        # by default the episode will terminate when all samples are labelled
        done = LalEnv._compute_is_terminal(self)
        # it also terminates when self.n_horizon datapoints were labelled
        if np.size(self.indeces_known) == self.n_horizon:
            done = True
        return done
    
    
class LalEnvTargetAccuracy(LalEnv): 
    """The LAL environment class where the episode lasts until a classifier reaches a predifined quality.

    This class inherits from LalEnv. 
    The reward is -1 at every step. 
    The terminal state is reached 
    when the predefined classificarion 
    quality is reached. Classification 
    quality is defined as a proportion 
    of the final quality (that is obtained 
    when all data is labelled).

    Attributes:
        tolerance_level: A float indicating what proportion of the maximum reachable score 
                         should be attained in order to terminate the episode.
        target_quality: A float indication the minimum required accuracy
                        after reaching which the episode is terminated.
    """
    
    def __init__(self, dataset, model, quality_method, tolerance_level=0.9):
        """Inits environment with its normal attributes + tolerance_level (proportion of quality to reach)."""
        LalEnv.__init__(self, dataset, model, quality_method)
        self.tolerance_level = tolerance_level
        self._set_target_quality()
    
    def _set_target_quality(self):
        """Sets the target accuracy according to the tolerance_level.

        This function computes the best reachable quality of the model
        on the full potential training data and sets the target_quality 
        as tolerance_level*max_qualtity.
        """
        best_model = clone(self.model)
        # train and avaluate the model on the full size of potential dataset
        best_model.fit(self.dataset.train_data, np.ravel(self.dataset.train_labels))
        test_prediction = best_model.predict(self.dataset.test_data)   
        max_quality = self.quality_method(self.dataset.test_labels, test_prediction)
        # the target_quality after which the episode stops is a proportion of the max quality
        self.target_quality = self.tolerance_level*max_quality
        
    def reset(self, n_start=2):
        """Resets the environment.
        
        First, do, what is done for the parent environment and then:
        4) The target quality for this experiment is set.

        Args:
            n_start: An integer indicating the size of annotated set at the beginning.
            
        Returns:
            the same as the parent class.
        """
        classifier_state, next_action_state = LalEnv.reset(self, n_start=n_start)
        self._set_target_quality()
        return classifier_state, next_action_state
       
    def _compute_reward(self):
        """Computes the reward.
        
        The reward is -1 in all states. In the terminal state, 
        the environment will stop issueing a negative reward 
        and suffering of annotating data is be ended.
        
        Returns:
            reward: A float: -1.
        """        
        reward = -1
        return reward
    
    
    def _compute_is_terminal(self):
        """Computes if the episode has reached the terminal state.
        
        The end of the episode is reached when the 
        classification accuracy reaches the predefined
        level.
        
        Returns:
            done: A boolean that indicates if the episode is finished.
        """
        new_score = self.episode_qualities[-1]
        # by default the episode will terminate when all samples are labelled
        done = LalEnv._compute_is_terminal(self)
        # it also terminates when a quality reaches a predefined level
        if new_score >= self.target_quality:
            done = True
        return done