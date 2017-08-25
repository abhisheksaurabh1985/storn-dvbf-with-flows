import os
import pickle
from matplotlib import pyplot as plt
plt.style.use('seaborn-deep')

import plots


"""
This script is for plotting the generative signals obtained from various models and comparing it with the actual signal.
It will generate as many plots as there are input signals. Each plot will have actual signal until a time step t, to be 
called 'cut-off time'. After t, generative signal obtained from different models will appear. Actual signal after time 
t, shall appear as a dotted curve.

For the inverted pendulum dataset, plot for rewards is blank. 
"""

# Step 1: Set up the input and the output directory.
input_dir = "./results_worth_saving/generative_samples/"
output_dir = "./results_worth_saving/generative_samples"

# Step 2: Get the file names of the pickled files corresponding to each model and actual data. Note that there are as
# many pickled files containing actual data as there are models. Though we'll be loading each of these, we'll be using
# only one of those. The prefix 'nf' stands for 'NoFlow'. IT DOES NOT STAND FOR 'NORMALIZING FLOWS'.

# Files containing actual and generative samples obtained from the 'NoFlow' model
fname_nf_x = os.path.join(input_dir, "nf_x_for_generative_sampling_for_comparison.pkl")  # Actual data
fname_nf_gs = os.path.join(input_dir, "nf_gs_samples_for_comparison.pkl")  # Generative samples

# Step 3: Load pickled data
nf_x = pickle.load(open(fname_nf_x, "rb"))
nf_gs = pickle.load(open(fname_nf_gs, "rb"))

# Step 4: Extract features from the data
nf_cos_actual = plots.helper_functions.sliceFrom3DTensor(nf_x, 0)
nf_sine_actual = plots.helper_functions.sliceFrom3DTensor(nf_x, 1)
nf_w_actual = plots.helper_functions.sliceFrom3DTensor(nf_x, 2)  # Angular velocity omega
nf_reward_actual = plots.helper_functions.sliceFrom3DTensor(nf_x, 3)
print nf_cos_actual.shape, nf_sine_actual.shape, nf_w_actual.shape, nf_reward_actual.shape

nf_cos_gs = plots.helper_functions.sliceFrom3DTensor(nf_gs, 0)
nf_sine_gs = plots.helper_functions.sliceFrom3DTensor(nf_gs, 1)
nf_w_gs = plots.helper_functions.sliceFrom3DTensor(nf_gs, 2)  # Angular velocity omega
nf_reward_gs = plots.helper_functions.sliceFrom3DTensor(nf_gs, 3)
print nf_cos_gs.shape, nf_sine_gs.shape, nf_w_gs.shape, nf_reward_gs.shape

# Prepare data for the plot function
num_time_steps = nf_gs.shape[0]  # Get a list of time steps
cut_off_time = 50
num_features = nf_gs.shape[2] - 1
flow_types = ["NF"]  # ["NF", "PF", "RF", "CPF"]
nf_actual_data = [nf_cos_actual, nf_sine_actual, nf_w_actual, nf_reward_actual]
nf_gs_data = [nf_cos_gs, nf_sine_gs, nf_w_gs, nf_reward_gs]

gs_data = [nf_gs_data]  # Add the data for generative signals from all the models in one list.




# Step 4: Plot four separate plots for each of the input features. Each plot will have four curves. One corresponding
# to the actual data until a certain time step t. After time step t, there will be curves corresponding to the signals
# obtained from different models. The actual signal should appear in dotted after t.
instances_to_plot = [0, 2, 4]


def plot_signals_and_generative_samples(n_time_steps, cut_off_time_step, n_feature, actual_signal, generative_signals,
                                        flow_types=None, output_dir=None, points_to_plot=[0, 2, 4]):

    nf_gen_signal = generative_signals[0]

    for each_flow_type in flow_types:
        for each_point in points_to_plot:
            for i in range(n_feature):
                plt.figure(each_flow_type + str(each_point) + str(i))
                plt.plot(range(cut_off_time_step), actual_signal[i][0:cut_off_time_step, each_point])
                plt.plot(range(cut_off_time_step, n_time_steps),
                         actual_signal[i][cut_off_time_step: n_time_steps, each_point], '--', linewidth=2)
                plt.plot(range(cut_off_time_step, n_time_steps),
                         nf_gen_signal[i][cut_off_time_step: n_time_steps, each_point], '--', linewidth=2)
                plt.axvline(x=cut_off_time_step, linewidth=0.5, color='black')
                plt.xlabel("Time steps")
                plt.ylabel("Input Feature Value")
                legend_labels = ["Actual", "Actual", "NF-GS"]
                plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
                plt.savefig(os.path.join(output_dir, str.lower(each_flow_type) + "_" + str(each_point) + "_" +
                                         str(i) + "_" + "gs.png"))


plot_signals_and_generative_samples(num_time_steps, cut_off_time, num_features, nf_actual_data, gs_data, flow_types,
                                    output_dir)




