from data_source import dataset_utils

import pickle
import numpy as np
import gym


class Datasets(object):
    pass


class Dataset(object):
    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[1]
        # pass

    @property
    def full_data(self):
        return self._data

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.seed(1)
            np.random.shuffle(idx)  # shuffle indexes
            self._data = self.full_data[:, idx, :]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.full_data[:, start:self._num_examples, :]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.seed(0)
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.full_data[:, idx0, :]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data_new_part = self._data[:, start:end, :]
            return np.concatenate((data_rest_part, data_new_part), axis=1)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[:, start:end, :]


# entry point
if __name__ == '__main__':
    # np.random.seed(0)
    env = gym.make('Pendulum-v0')
    n_samples = 1000
    n_timesteps = 100
    test_size = 0.2 # Percentage of test data
    learned_reward = True  # is the reward handled as observation?
    include_angular_velocity = False
    # # 4 dimensions and the control signal combined would be the input variable.
    # X.shape: (100, 1000, 4); U.shape:(100, 1000,1). The 4 dimensions correspond to
    # cosine and sine of angle alpha, angular velocity and reward.
    # U is the one dimensional control signal at each time step.
    X, U = dataset_utils.rollout(env, n_samples, n_timesteps, learned_reward=learned_reward, fn_action=None)
    if include_angular_velocity:
        XU = np.concatenate((X, U), -1)
        x_train, x_test = np.split(XU, [int(.8*XU.shape[1])], axis=1)

        datasets = Datasets()
        datasets.train = Dataset(x_train)
        datasets.test = Dataset(x_test)

        with open('./pickled_data/datasets.pkl', "wb") as f:
            pickle.dump(datasets, f)
    else:
        X = np.delete(X, [2], axis=2)
        XU = np.concatenate((X, U), -1)
        x_train, x_test = np.split(XU, [int(.8*XU.shape[1])], axis=1)

        datasets = Datasets()
        datasets.train = Dataset(x_train)
        datasets.test = Dataset(x_test)

        with open('./pickled_data/datasets_sans_av.pkl', "wb") as f:
            pickle.dump(datasets, f)

