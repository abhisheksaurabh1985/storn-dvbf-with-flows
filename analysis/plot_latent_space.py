from data_source.data_set import Datasets, Dataset
from data_source import dataset_utils

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from mpl_toolkits.mplot3d import Axes3D


def plot_latent_space(time_steps, batch_size, data, dir_output):
    """

    :param time_steps: time steps as a list from o through t.
    :param batch_size:
    :param data: (t,b,d) data reshaped as (-1,2).
    :param dir_output: Output directory
    :return:
    """
    # Step 3: Plot data
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=15., azim=30.)

    # time_steps = range(latent_data.shape[1])

    # Plot the first instance in the batch so as to trace its path in the latent space.
    ax.plot(data[0:100, 0], data[0:100, 1], np.array(time_steps),
            label='posterior walk',
            color='k', lw=3)

    # Plot all the points
    ts_for_all_data_points = np.array(time_steps * batch_size)  # Time steps for all instances in the batch.

    # Give a color based on the cosine of the angle.
    ax.scatter(data[:, 0], data[:, 1], ts_for_all_data_points,
               c=data[:, 0], cmap='jet')
    ax.set_xlabel('First Latent Dimension, $\mathbf{z_1}$', fontsize=10, fontweight='bold')
    ax.set_ylabel('Second Latent Dimension, $\mathbf{z_2}$', fontsize=10, fontweight='bold')
    ax.set_zlabel('Time steps', fontsize=10, fontweight='bold')
    ax.set_title('2D Latent Space of the Inverted Pendulum Dataset', fontsize=14, fontweight='bold')
    fig.savefig(os.path.join(dir_output, "latent_space.png"))

if __name__ == '__main__':
    # Step 1: Set up the input and the output directory.

    bs = 20

    input_dir = ".././pickled_data/datasets.pkl"
    output_dir = ".././output/"
    datasets = pickle.load(open(".././pickled_data/datasets.pkl", "rb"))

    latent_data = datasets.train.next_batch(bs)  # This is (t,b,5).
    latent_data = latent_data[:, :, 0:2]
    latent_data = np.transpose(latent_data, [1, 0, 2])  # Reshape as (b,t,d).
    print "latent_data shape", latent_data.shape

    # Step 2: Reshape the data as (-1,2).
    reshaped_latent_data = np.reshape(latent_data, (-1, 2))  # (2000,2)
    print "reshaped_latent_data shape", reshaped_latent_data.shape

    ts = range(latent_data.shape[1])  # Get time steps

    plot_latent_space(ts, bs, reshaped_latent_data, output_dir)
