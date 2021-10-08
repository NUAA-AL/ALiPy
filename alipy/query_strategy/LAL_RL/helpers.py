import numpy as np

class Minibatch:
    """Minibatch class that helps for gradient decent training.

    Attributes:
        classifier_state: A numpy.ndarray of size batch_size x #classifier features
            characterising the state of classifier at the sampled iterations
        action_state: A numpy.ndarray of size batch_size x #action features
            characterizing the action that was taken at the sampled iterations
        reward: A numpy.ndarray of size batch_size
        next_classifier_state: A numpy.ndarray of size batch_size x #classifier features
        next_action_state: A list of size batch_size of numpy.ndarrays 
            characterizing the possible actions that were available at the sampled iterations
        terminal: A numpy.ndarray of size batch_size of booleans indicating if the iteration was terminal
        indeces: A numpy.ndarray of size batch_size that contains indeces of samples iterations in the replay buffer
    """
    def __init__(self, classifier_state, action_state, reward, next_classifier_state, next_action_state, terminal, indeces):
        """Inits the Minibatch object and initialises the attributes with given values."""
        self.classifier_state = classifier_state
        self.action_state = action_state
        self.reward = reward
        self.next_classifier_state = next_classifier_state
        self.next_action_state = next_action_state
        self.terminal = terminal
        self.indeces = indeces
    
    
class ReplayBuffer:
    """Replay Buffer is used to store the transactions from episodes.

    Attributes:
        buffer_size: An interger indicating the maximum number of transaction to be stored in the replay buffer.
        n: An interger, the maximum index to be used for sampling. It is useful when the buffer is not filled in fully.
           It grows from 0 till the buffer_size-1 and then stops changing.
        write_index: An integer, the index where the next transaction should be written. 
           Goes from 0 till the buffer_size-1 and then starts from 0 again.
        max_td_error: A float used to initialize the td error of newly added samples.
        prior_exp: A float that is used for turning the td error into a probability to be sampled. 
        all_classifier_state: A numpy.ndarray of size batch_size x #classifier features
           characterising the state of classifier at the sampled iterations.
        all_action_states: A numpy.ndarray of size batch_size x #action features
           characterizing the action that was taken at the sampled iterations.
        all_rewards: A numpy.ndarray of size batch_size.
        all_next_classifier_states: A numpy.ndarray of size batch_size x #classifier features
        all_next_action_state: A list of size batch_size of numpy.ndarrays.
           characterizing the possible actions that were available at the sampled iterations.
        all_terminals: A numpy.ndarray of size batch_size of booleans indicating if the iteration was terminal.
        all_td_errors: A numpy.ndarray of size batch_size with td errors of transactions 
           when each of them was used in a gradient update.
        max_td_error: A float with the highest (absolute) value of td error from all transactions stored in the buffer.
    """
    
    def __init__(self, buffer_size=1e4, prior_exp=0.5):
        """Inits a few attributes with 0 or the default values."""
        self.buffer_size = int(buffer_size)
        self.n = 0
        self.write_index = 0
        self.max_td_error = 1000.0
        self.prior_exp = prior_exp
    
    def _init_nparray(self, classifier_state, action_state, reward, next_classifier_state, next_action_state, terminal):
        """Initialize numpy arrays of all_xxx attributes to one transaction repeated buffer_size times."""
        self.all_classifier_states = np.array([classifier_state] * self.buffer_size)
        self.all_action_state = np.array([action_state] * self.buffer_size)
        self.all_rewards = np.array([reward] * self.buffer_size)
        self.all_next_classifier_states = np.array([next_classifier_state] * self.buffer_size)
        self.all_next_action_states = [next_action_state] * self.buffer_size
        self.all_terminals = np.array([terminal] * self.buffer_size)
        self.all_td_errors = np.array([self.max_td_error] * self.buffer_size)
        # set the counters to 1 as one transaction is stored
        self.n = 1
        self.write_index = 1
  
    def store_transition(self, classifier_state, action_state, reward, next_classifier_state, next_action_state, terminal):
        """Add a new transaction to a replay buffer."""
        # If buffer arrays not yet initialized, initialize it
        if self.n == 0:
            self._init_nparray(classifier_state, action_state, reward, next_classifier_state, next_action_state, terminal)
            return
        # write a tansaction at a write_index position
        self.all_classifier_states[self.write_index] = classifier_state
        self.all_action_state[self.write_index] = action_state
        self.all_rewards[self.write_index] = reward
        self.all_next_classifier_states[self.write_index] = next_classifier_state
        self.all_next_action_states[self.write_index] = next_action_state
        self.all_terminals[self.write_index] = terminal
        self.all_td_errors[self.write_index] = self.max_td_error
        # keep track of the index for writing
        self.write_index += 1
        if self.write_index >= self.buffer_size:
            self.write_index = 0
        # Keep track of the max index to be used for sampling.
        if self.n < self.buffer_size:
            self.n += 1

    def sample_minibatch(self, batch_size=32):
        """Sample a new minibatch from replay buffer.
        
        Args:
            batch_size: An integer indicating how many transactions to be sampled from a replay buffer.
            
        Returns:
            minibatch: An object of class Minibatch with sampled transactions.
        """
        # Get td error of samples that were written in the buffer
        td_errors_to_consider = self.all_td_errors[:self.n]
        # Scale and normalize the td error to turn it into a probability for sampling
        p = np.power(td_errors_to_consider, self.prior_exp) / np.sum(np.power(td_errors_to_consider, self.prior_exp))
        # choose indeces to sample according to the computed probability: 
        # the higher the td error is, the more likely it is that the sample will be selected
        # first check if the number of non-zero elements in p is smaller than the batch_size
        non_zero = np.count_nonzero(p)
        if non_zero < batch_size <= self.n:
            minibatch_indices = np.random.choice(range(self.n), size=non_zero, replace=False, p=p)
            # add the missing elements
            missing_elements = batch_size - non_zero
            while missing_elements > 0:
                num = np.random.choice(range(self.n))
                if not num in minibatch_indices:
                    minibatch_indices = np.concatenate((minibatch_indices, [num]))
                    missing_elements -= 1
        else:
            minibatch_indices = np.random.choice(range(self.n), size=batch_size, replace=False, p=p)
        minibatch = Minibatch(
            self.all_classifier_states[minibatch_indices],
            self.all_action_state[minibatch_indices],
            self.all_rewards[minibatch_indices],
            self.all_next_classifier_states[minibatch_indices],
            [self.all_next_action_states[i] for i in minibatch_indices],
            self.all_terminals[minibatch_indices],
            minibatch_indices,
        )
        return minibatch
    
    def update_td_errors(self, td_errors, indeces):
        """Updates td_errors in replay buffer.
        
        After a gradient step was made, we need to updates 
        td errors to recently calculated errors.
        
        Args:
            td_errors: A numpy array with new td errors.
            indeces: A numpy array with indeces of points which td errors should be updated.
        """
        # set the values for prioritized replay to the most recent td errors
        self.all_td_errors[indeces] = np.ravel(np.absolute(td_errors))
        # find the max error from the replay buffer that will be used as a default value for new transactions
        self.max_td_error = np.max(self.all_td_errors)