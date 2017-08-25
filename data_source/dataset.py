import pickle
import numpy as np
import gym

from data_source import dataset_utils


class Datasets(object):
    pass


class Dataset(object):
    def __init__(self, features):
        # assert features.shape[0] == labels.shape[0], ("features.shape: %s labels.shape: %s" % (features.shape, labels.shape))
        self._num_examples = features.shape[1]

        features = features.astype(np.float32)
        # features = np.multiply(features - 130.0, 1.0 / 70.0) # [130.0 - 200.0] -> [0 - 1]
        self._features = features
        # self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def full_data(self):
        return self._features

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[:,perm,]
            # self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[:,start:end,:] # , self._labels[start:end]


def corresponding_shuffle(data):
    """
    # Shuffle data and its label in association
    :param data:
    :return:
    """
    if len(data.shape) == 3:
        random_indices = np.random.permutation(data.shape[1])
        _data = np.zeros_like(data)
        for i,j in enumerate(random_indices):
            _data[:,i] = data[:,j,:]
    elif len(data.shape) == 2:
        random_indices = np.random.permutation(len(data))
        _data = np.zeros_like(data)
        for i,j in enumerate(random_indices):
            _data[i] = data[j]
    return _data, random_indices        


def data_sanity_check_post_shuffling(data, shuffled_data, random_indices):
    result = [] 
    for i, j in enumerate(random_indices):
        result.append((shuffled_data[:, i, :] == data[:, j, :]).all())
    # Look for the occurrence of False in the list 
    if all(result):
        print "Data shuffled correctly. Sanity check passed."
    else:
        print "Data NOT shuffled correctly. Check script."
    return result    
        

# save dataset as a pickle file
def save_as_pickle(filename, dataset):
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)


# entry point
if __name__ == '__main__':

    np.random.seed(0)

    datasets = Datasets()
    env = gym.make('Pendulum-v0')
    n_samples = 1000
    n_timesteps = 100
    test_size = 0.2 # Percentage of test data
    # is the reward handled as observation?
    learned_reward = True
    
    X, U = dataset_utils.rollout(env, n_samples, n_timesteps, learned_reward=learned_reward, fn_action=None)
    # X_mean = X.reshape((-1, X.shape[2])).mean(0)
    # X = X - X_mean
    # X_std = X.reshape((-1, X.shape[2])).std(0)
    # X = X / X_std
    # # 4 dimensions and the control signal combined would be the input variable.
    # X.shape: (100, 1000, 4); U.shape:(100, 1000,1). The 4 dimensions correspond to
    # cosine and sine of angle alpha, angular velocity and reward. 
    # U is the one dimensional control signal at each time step. 
    XU = np.concatenate((X, U), -1)

    # Shuffle data for creating mini-batches
    shuffled_data, random_indices = corresponding_shuffle(XU)
    # Check if data was shuffled correctly
    data_sanity_check_post_shuffling(XU, shuffled_data, random_indices)

    # Split into train and test data
#    x_train, x_test, _, _ = train_test_split(shuffled_data, test_size = test_size) 
#    print x_train.shape, x_test.shape        

    x_train, x_test = np.split(shuffled_data, 
                               [int(.8*shuffled_data.shape[1])], axis=1)
    
    # Create dataset
    datasets.train = Dataset(x_train)
    datasets.test = Dataset(x_test)

    # Save as a pickle file
    save_as_pickle('./pickled_data/XU.pkl', XU)
    save_as_pickle('./pickled_data/shuffled_data.pkl', shuffled_data)
    save_as_pickle('./pickled_data/datasets.pkl', datasets)
