import copy
from .envs import LalEnvTargetAccuracy
from .datasets import DatasetUCI
from .helpers import ReplayBuffer
from .Agent import Agent, Net
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy
import collections
from ..base import BaseIndexQuery

from tqdm import tqdm

import torch


class LAL_RL_StrategyLearner:
    """Reference Paper:
        Ksenia Konyushkova, Raphael Sznitman, Pascal Fua, 2018.
        Discovering General-Purpose Active Learning Strategies.
        https://arxiv.org/abs/1810.04114

    The implementation is referred to
    https://github.com/ksenia-konyushkova/LAL-RL
    """
    def __init__(self, path, possible_dataset_names, n_state_estimation=30, size=-1, subset=-1,
                 quality_method=metrics.accuracy_score, tolerance_level=0.98, model=None,
                 replay_buffer_size = 1e4, prioritized_replay_exponent = 3) :
        """
        path: the directory that contains the datasets
        possible_dataset_names = the name of the datasets that will be used for training
        n_state_estimation: how many datapoints are used to represent a state
        size: An integer indicating the size of training dataset to sample, if -1 use all data
        subset: An integer indicating what subset of data to use. 0: even, 1: odd, -1: all datapoints
        quality_method: the measure that will be used for the target quality
        tolerance_level: the ratio of the target quality that the agent has to achieve to end an episode
        model: the model that is used to make predictions on the datasets, should implement fit, predict and predict_proba
        replay_buffer_size: An interger indicating the maximum number of transaction to be stored in the replay buffer
        prioritized_replay_exponent: A float that is used for turning the td error into a probability to be sampled
        """
        dataset = DatasetUCI(possible_dataset_names, n_state_estimation=n_state_estimation, subset=subset,
                             size=size,path=path)
        if model == None:
            model = LogisticRegression()
        self.env = LalEnvTargetAccuracy(dataset, model, quality_method=quality_method,
                                        tolerance_level=tolerance_level)
        self.n_state_estimation = n_state_estimation
        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size, prior_exp=prioritized_replay_exponent)


    def train_query_strategy(self, saving_path, file_name, warm_start_episodes=100, nn_updates_per_warm_start=100,
                learning_rate=1e-4, batch_size=32, gamma=1, target_copy_factor=0.01,
                training_iterations=1000, episodes_per_iteration=10, updates_per_iteration=60,
                epsilon_start=1, epsilon_end=0, epsilon_step=1000, device=None, verbose=2):
        """
        saving_path: the directory where the learnt strategy will be saved
        file_name: the file name for the learnt strategy
        warm_start_episodes: the number of warm start episodes that will be performed before the actual training
        nn_updates_per_warm_start: the number of q-network updates after the warm-start-episodes
        learning_rate: the learning rate of the deep q-network
        batch_size: the size of the batches that will be sampled from replay memory for one q-network update
        gamma: the discount factor in q-learning
        target_copy_factor: the factor for copying the weights of the estimator to the target estimator
        training_iterations: the amount of training iterations
        episodes_per_iteration: the amount of episodes in one training iteration
        updates_per_iteration: the number of q-network updates that are performed at the end of an iteration
        epsilon_start: the start value of epsilon for the epsilon greedy strategy
        epsilon_end: the end value of epsilon for the epsilon greedy strategy
        epsilon_step: the number of iterations it takes for the epsilon value to decay from start value to end value
        device: pytorch device that will be used for the computations
        verbose: 3 - progessbar for warmstart episodes, iterations, episodes and steps in the environment
                 2 - progessbar for warmstart episodes, iterations and episodes
                 1 - just one progessbar for the iterations
                 0 - no progressbars
        """
        if verbose not in [0,1,2,3]:
            raise ValueError("Verbose must be 0, 1, 2 or 3")
        self.saving_path = saving_path + "/" + file_name
        self.batch_size = batch_size
        self.training_iterations = training_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.updates_per_iteration = updates_per_iteration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = epsilon_step
        self.verbose = verbose

        bias_average = self.run_warm_start_episodes(warm_start_episodes)
        self.agent = Agent(self.n_state_estimation, learning_rate, batch_size, bias_average,
               target_copy_factor, gamma, device)
        self.train_agent(nn_updates_per_warm_start)
        self.run_training_iterations()


    def continue_training(self, saving_path, file_name, net_path, target_net_path=None, learning_rate=1e-4, batch_size=32,
                gamma=1, target_copy_factor=0.01, training_iterations=1000, episodes_per_iteration=10, updates_per_iteration=60,
                epsilon_start=1, epsilon_end=0, epsilon_step=1000, device=None, verbose=2):
        """
        net_path: the path to the q-network that has already been trained and shall now be further trained
        target_net_path: the corresponding target net if None then a copy of the net will be used as target net

        the other parameters are exactly the same as in train_query_strategy
        """
        if verbose not in [0,1,2,3]:
            raise ValueError("Verbose must be 0, 1, 2 or 3")
        state_dict = torch.load(net_path, map_location=device)

        if target_net_path != None:
            target_state_dict = torch.load(target_net_path, map_location=device)
        else:
            target_state_dict = copy.deepcopy(state_dict)

        # test if given n_state_estimation matches the one of the loaded state_dict
        if self.n_state_estimation != state_dict[list(state_dict.keys())[0]].size(1):
            raise ValueError("given n_state_estimation doesn't match the one of the loaded state_dict")
        # test if n_state_estimation of net and target net are the same
        if state_dict[list(state_dict.keys())[0]].size(1) != state_dict[list(target_state_dict.keys())[0]].size(1):
            raise ValueError("n_state_estimation of net and target net are not the same")

        self.saving_path = saving_path + "/" + file_name
        self.batch_size = batch_size
        self.training_iterations = training_iterations
        self.episodes_per_iteration = episodes_per_iteration
        self.updates_per_iteration = updates_per_iteration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_step = epsilon_step

        self.agent = Agent(self.n_state_estimation, learning_rate, batch_size, 0,
               target_copy_factor, gamma, device)
        self.agent.net.load_state_dict(state_dict)
        self.agent.target_net.load_state_dict(target_state_dict)
        self.run_training_iterations()


    def run_warm_start_episodes(self, n_episodes):
        # create function depending on verbose level
        if self.verbose >= 2:
            p_bar = tqdm(total=n_episodes, desc="Warmstart episodes", leave=False)
            def update():
                p_bar.update()
            def close():
                p_bar.close()
        else:
            update = lambda *x : None
            close = lambda *x : None

        # Keep track of episode duration to compute average
        episode_durations = []
        for _ in range(n_episodes):
            # Reset the environment to start a new episode
            # classifier_state contains vector representation of state of the environment (depends on classifier)
            # next_action_state contains vector representations of all actions available to be taken at the next step
            classifier_state, next_action_state = self.env.reset()
            terminal = False
            episode_duration = 0
            # before we reach a terminal state, make steps
            while not terminal:
                # Choose a random action
                action = np.random.randint(0, self.env.n_actions)
                # taken_action_state is a vector corresponding to a taken action
                taken_action_state = next_action_state[:,action]
                next_classifier_state, next_action_state, reward, terminal = self.env.step(action)
                # Store the transition in the replay buffer
                self.replay_buffer.store_transition(classifier_state, 
                                            taken_action_state, 
                                            reward, next_classifier_state, 
                                            next_action_state, terminal)
                # Get ready for next step
                classifier_state = next_classifier_state
                episode_duration += 1 
            episode_durations.append(episode_duration)
            update()
        # compute the average episode duration of episodes generated during the warm start procedure
        av_episode_duration = np.mean(episode_durations)
        close()

        return -av_episode_duration/2


    def train_agent(self, n_of_updates):
        # check if there are enough experiences in replay memory
        if self.replay_buffer.n < self.batch_size:
            return
        # create function depending on verbose level
        if self.verbose >= 2:
            p_bar = tqdm(total=n_of_updates, desc="Train q-net", leave=False)
            def update():
                p_bar.update()
            def close():
                p_bar.close()
        else:
            update = lambda *x : None
            close = lambda *x : None
        for _ in range(n_of_updates):
            # Sample a batch from the replay buffer proportionally to the probability of sampling.
            minibatch = self.replay_buffer.sample_minibatch(self.batch_size)
            # Use batch to train an agent. Keep track of temporal difference errors during training.
            td_error = self.agent.train(minibatch)
            # Update probabilities of sampling each datapoint proportionally to the error.
            self.replay_buffer.update_td_errors(td_error, minibatch.indeces)
            update()
        close()


    def run_training_iterations(self):
        # create function depending on verbose level
        if self.verbose >= 1:
            p_bar_iter = tqdm(total=self.training_iterations, desc="Train iterations", leave=(self.verbose > 2))
            def update_iter():
                p_bar_iter.update()
            def close_iter():
                p_bar_iter.close()
        else:
            update_iter = lambda *x : None
            close_iter = lambda *x : None

        for iteration in range(self.training_iterations):
            # GENERATE NEW EPISODES
            # Compute epsilon value according to the schedule.
            epsilon = max(self.epsilon_end, self.epsilon_start-iteration*(self.epsilon_start-self.epsilon_end)/self.epsilon_step)
            
            # create function depending on verbose level
            if self.verbose >= 2:
                p_bar_episode = tqdm(total=self.episodes_per_iteration, desc="Episodes", leave=False)
                def update_episode():
                    p_bar_episode.update()
                def close_episode():
                    p_bar_episode.close()
            else:
                update_episode = lambda *x : None
                close_episode = lambda *x : None

            # Simulate training episodes.
            for _ in range(self.episodes_per_iteration):
                # Reset the environment to start a new episode.
                classifier_state, next_action_state = self.env.reset()
                terminal = False
                max_steps = len(self.env.dataset.train_labels)

                # create function depending on verbose level
                if self.verbose >= 3:
                    p_bar_steps = tqdm(total=max_steps, desc=self.env.dataset.dataset_name, leave=False)
                    def update_steps():
                        p_bar_steps.update()
                    def close_steps():
                        p_bar_steps.close()
                else:
                    update_steps = lambda *x : None
                    close_steps = lambda *x : None
                # Run an episode.
                while not terminal:
                    # Let an agent choose an action or with epsilon probability, take a random action.
                    if np.random.ranf() < epsilon: 
                        action = np.random.randint(0, self.env.n_actions)
                    else:
                        action = self.agent.get_action(classifier_state, next_action_state)
                    
                    # taken_action_state is a vector that corresponds to a taken action
                    taken_action_state = next_action_state[:,action]
                    # Make another step.
                    next_classifier_state, next_action_state, reward, terminal = self.env.step(action)
                    # Store a step in replay buffer
                    self.replay_buffer.store_transition(classifier_state, 
                                                taken_action_state, 
                                                reward, 
                                                next_classifier_state, 
                                                next_action_state, 
                                                terminal)
                    # Change a state of environment.
                    classifier_state = next_classifier_state
                    update_steps()
                close_steps()
                update_episode()
            close_episode()
            # NEURAL NETWORK UPDATES
            self.train_agent(self.updates_per_iteration)
            update_iter()

        self.agent.save_net(self.saving_path)
        self.agent.save_target_net(self.saving_path)
        close_iter()



class QueryInstanceLAL_RL(BaseIndexQuery):
    """This class uses a strategy that was learnt by LAL_RL_StrategyLearner.

    Parameters
    ----------
    X: 2D array,
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    y: array-like,
        Label matrix of the whole dataset. It is a reference which will not use additional memory.

    model_path: file-like object or string or os.PathLike object,
        state_dict of the trained strategy

    n_state_estimation: int, optional (default=None)
        number of datapoints used by the strategy to build the state, if None is provided an inference is attempted

    pre_batch: int, optional (default=128)
        batch size that is used when predicting with the learnt strategy

    device: torch.device, optional (default=None)
        the pytorch device used for the calculations
    
    """
    def __init__(self, X, y, model_path, n_state_estimation=None, pred_batch=128, device=None, **kwargs):
        super(QueryInstanceLAL_RL, self).__init__(X, y)
        state_dict = torch.load(model_path, map_location=device)
        self.pred_batch = pred_batch
        self.device = device
        if n_state_estimation == None:
            self.n_state_estimation = state_dict[list(state_dict.keys())[0]].size(1)
        else:
            self.n_state_estimation = n_state_estimation
        self.net = Net(self.n_state_estimation,0)
        self.net.load_state_dict(state_dict)
        self.net.to(device)
        self.net.eval()

    def select(self, label_index, unlabel_index, model=None, batch_size=1, **kwargs):
        # copy label_index and unlabel_index
        label_index_copy = copy.deepcopy(label_index)
        unlabel_index_copy = copy.deepcopy(unlabel_index)
        assert (batch_size > 0)
        assert (isinstance(unlabel_index_copy, collections.abc.Iterable))
        assert (isinstance(label_index_copy, collections.abc.Iterable))
        if len(unlabel_index_copy) <= batch_size:
            return unlabel_index_copy
        assert len(unlabel_index_copy) + len(label_index_copy) >= self.n_state_estimation
        unlabel_index_copy = np.asarray(unlabel_index_copy)

        # initialize the model and train it if necessary
        if model == None:
            model = LogisticRegression()
            model.fit(self.X[label_index_copy], self.y[label_index_copy])
        
        # set aside some unlabeled data for the state representation, the data is removed from the unlabel_index
        if len(unlabel_index_copy) >= self.n_state_estimation + batch_size:
            chosen_indices = np.random.choice(len(unlabel_index_copy), size=self.n_state_estimation, replace=False)
            state_indices = unlabel_index_copy[chosen_indices]
            unlabel_index_copy = unlabel_index_copy[np.array([x for x in range(len(unlabel_index_copy)) if x not in chosen_indices])]

        # if there isn't enough data then also the label_index is used and the data is not removed from the indicies
        else:
            state_indices = np.random.choice(np.concatenate((np.array(label_index_copy), np.array(unlabel_index_copy))), 
                    size=self.n_state_estimation, replace=False)

        # create the state
        predictions = model.predict_proba(self.X[state_indices])[:,0]
        predictions = np.array(predictions)
        idx = np.argsort(predictions)
        state = predictions[idx]

        #create the actions
        a1 = model.predict_proba(self.X[unlabel_index_copy])[:,0]

        # calculate distances
        data = self.X[np.concatenate((label_index_copy,unlabel_index_copy),axis=0)]
        distances = scipy.spatial.distance.pdist(data, metric='cosine')
        distances = scipy.spatial.distance.squareform(distances)
        indeces_known = np.arange(len(label_index_copy))
        indeces_unknown = np.arange(len(label_index_copy), len(label_index_copy)+len(unlabel_index_copy))
        a2 = np.mean(distances[indeces_unknown,:][:,indeces_unknown],axis=0)
        a3 = np.mean(distances[indeces_known,:][:,indeces_unknown],axis=0)

        actions = np.concatenate(([a1], [a2], [a3]), axis=0).transpose()

        # calculate the q-values according to the q-network
        # first transform the state and actions for the network
        state = np.repeat([state], actions.shape[0], axis=0)
        state_actions = np.concatenate((state,actions),axis=1)
        input_tensor = torch.tensor(state_actions, dtype=torch.float, device=self.device)

        # get the prediction from the network
        pred = self.net(input_tensor[:self.pred_batch])
        for i in range(self.pred_batch, input_tensor.size(0), self.pred_batch):
            pred = torch.cat((pred, self.net(input_tensor[i:i+self.pred_batch])))
        pred = pred.flatten()

        # sort the actions with respect to their q-value
        idx = pred.argsort(descending=True)
        idx = idx[:batch_size].detach().cpu().numpy()

        # return the correspoding indeces from the unlabeld index
        return unlabel_index_copy[idx]