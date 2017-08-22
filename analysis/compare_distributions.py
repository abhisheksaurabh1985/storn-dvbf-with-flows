import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


# Output directory
output_dir = "./results_worth_saving/data_dist_plot"

# Specify path to the pickled file
data_dir = "./results_worth_saving/data_dist_plot"
file_recons_nf = os.path.join(data_dir, 'nf_saved_vars.pickle')
file_recons_pf = os.path.join(data_dir, 'pf_saved_vars.pickle')
file_recons_cp = os.path.join(data_dir, 'cp_saved_vars.pickle')
file_recons_rf = os.path.join(data_dir, 'rf_saved_vars.pickle')


with open(file_recons_nf) as f:
    x_sample_nf, x_recons_nf, _, _, _ = pickle.load(f)

with open(file_recons_pf) as f:
    x_sample_pf, x_recons_pf, _, _, _ = pickle.load(f)

with open(file_recons_cp) as f:
    x_sample_cp, x_recons_cp, _, _, _ = pickle.load(f)

with open(file_recons_rf) as f:
    x_sample_rf, x_recons_rf, _, _, _ = pickle.load(f)

cos_actual = x_sample_nf[:, 0, 0].reshape((-1,))
sin_actual = x_sample_nf[:, 0, 1].reshape((-1,))
av_actual = x_sample_nf[:, 0, 2].reshape((-1,))  # Angular velocity
reward_actual = x_sample_nf[:, 0, 2].reshape((-1,))

cos_nf = x_recons_nf[:, 0, 0].reshape((-1,))
sin_nf = x_recons_nf[:, 0, 1].reshape((-1,))
av_nf = x_recons_nf[:, 0, 2].reshape((-1,))  # Angular velocity
reward_nf = x_recons_nf[:, 0, 3].reshape((-1,))

cos_pf = x_recons_pf[:, 0, 0].reshape((-1,))
sin_pf = x_recons_pf[:, 0, 1].reshape((-1,))
av_pf = x_recons_pf[:, 0, 2].reshape((-1,))  # Angular velocity
reward_pf = x_recons_pf[:, 0, 3].reshape((-1,))

cos_cp = x_recons_cp[:, 0, 0].reshape((-1,))
sin_cp = x_recons_cp[:, 0, 1].reshape((-1,))
av_cp = x_recons_cp[:, 0, 2].reshape((-1,))  # Angular velocity
reward_cp = x_recons_cp[:, 0, 3].reshape((-1,))

cos_rf = x_recons_rf[:, 0, 0].reshape((-1,))
sin_rf = x_recons_rf[:, 0, 1].reshape((-1,))
av_rf = x_recons_rf[:, 0, 2].reshape((-1,))  # Angular velocity
reward_rf = x_recons_rf[:, 0, 3].reshape((-1,))

plt.figure(0)
cos = np.vstack([cos_actual, cos_nf, cos_pf, cos_cp, cos_rf]).T
bins_cos = np.linspace(-1, 1, 10)
plt.hist(cos, bins_cos, alpha=0.7, label=['Actual', 'NoFlow', 'Planar', 'ConvPlanar', 'Radial'])
plt.legend(loc='upper right')
plt.title('Distribution of Cosine in Actual and Reconstructed Signal')
plt.savefig(os.path.join(output_dir, "dist_cos.png"))

plt.figure(1)
sin = np.vstack([sin_actual, sin_nf, sin_pf, sin_cp, sin_rf]).T
bins_sin = np.linspace(-1, 1, 10)
plt.hist(sin, bins_sin, alpha=0.7, label=['Actual', 'NoFlow', 'Planar', 'ConvPlanar', 'Radial'])
plt.legend(loc='upper right')
plt.title('Distribution of Sine in Actual and Reconstructed Signal')
plt.savefig(os.path.join(output_dir, "dist_sin.png"))

plt.figure(2)
av = np.vstack([av_actual, av_nf, av_pf, av_cp, av_rf]).T
bins_av = np.linspace(-8, 8, 10)
plt.hist(av, bins_av, alpha=0.7, label=['Actual', 'NoFlow', 'Planar', 'ConvPlanar', 'Radial'])
plt.legend(loc='upper right')
plt.title('Distribution of Average Velocity in Actual and Reconstructed Signal')
plt.savefig(os.path.join(output_dir, "dist_av.png"))

plt.figure(3)
reward = np.vstack([reward_actual, reward_nf, reward_pf, reward_cp, reward_rf]).T
bins_reward = np.linspace(-16, 0, 10)
plt.hist(reward, bins_reward, alpha=0.7, label=['Actual', 'NoFlow', 'Planar', 'ConvPlanar', 'Radial'])
plt.legend(loc='upper right')
plt.title('Distribution of Rewards in Actual and Reconstructed Signal')
plt.savefig(os.path.join(output_dir, "dist_reward.png"))

