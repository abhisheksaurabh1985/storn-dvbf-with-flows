# Module level imports
import plots

# Python standard and third party libraries
import os
import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.style.use('seaborn-deep')


def plot_signals_and_generative_samples(n_time_steps, cut_off_time_step, n_feature, actual_signal, generative_signals,
                                        output_dir=None, points_to_plot=[0, 2, 4, 6, 8, 10],
                                        plot_titles=["Cosine", "Sine", "Angular Velocity", "Reward"]):
    """
    This function is for plotting the generative signals obtained from various models and comparing it with the actual
    signal. It will generate as many plots as there are input signals. Each plot will have actual signal until a time
    step 't', to be called 'cut-off time'. After t, generative signal obtained from different models will appear.
    Actual signal after the cut-off time, shall appear as a dotted curve.

    Naming convention for the figures saved: dataInstanceInBatch_featureNumber.png

    :param n_time_steps: Number of time steps as integer.
    :param cut_off_time_step: Time step from which we want to start comparing the generative samples, as integer.
    :param n_feature: Number of features in input data, as integer.
    :param actual_signal: List containing the values of cos, sine, angular velocity and reward as a 1D numpy array.
    :param generative_signals: Nested list. Inner list contains a list of values of cos, sine, angular velocity and
    reward as 1D numpy array.
    :param output_dir: Output directory
    :param points_to_plot: List containing the index of the data points in the batch for which plots are desired.
    :param plot_titles: List containing the plot titles as string.
    :return: None
    """

    nf_gen_signal = generative_signals[0]
    pf_gen_signal = generative_signals[1]
    rf_gen_signal = generative_signals[2]

    for each_point in points_to_plot:
        for i in range(n_feature):
            plt.figure(str(each_point) + str(i))
            plt.plot(range(n_time_steps), actual_signal[i][:, each_point])
            # plt.plot(range(cut_off_time_step), actual_signal[i][0:cut_off_time_step, each_point])
            # plt.plot(range(cut_off_time_step, n_time_steps),
            #          actual_signal[i][cut_off_time_step: n_time_steps, each_point], '--', linewidth=2)
            plt.plot(range(cut_off_time_step, n_time_steps),
                     nf_gen_signal[i][cut_off_time_step: n_time_steps, each_point], '--', linewidth=2)
            plt.plot(range(cut_off_time_step, n_time_steps),
                     pf_gen_signal[i][cut_off_time_step: n_time_steps, each_point], '--', linewidth=2)
            plt.plot(range(cut_off_time_step, n_time_steps),
                     rf_gen_signal[i][cut_off_time_step: n_time_steps, each_point], '--', linewidth=2)
            plt.axvline(x=cut_off_time_step, linewidth=0.5, color='black')
            plt.xlabel("Time steps")
            plt.ylabel("Input Feature Values")
            legend_labels = ["Actual", "NF-GS", "PF-GS", "RF-GS"]
            plt.legend(legend_labels, bbox_to_anchor=(0.01, 0.85, 0.35, 0.5), loc=4,
                       ncol=2, mode="expand", borderaxespad=0., title="Legend", prop={'size': 8})
            # plt.legend(legend_labels, bbox_to_anchor=(0, 0.85, 1, 1), loc=4,
            #            ncol=3, mode="expand", borderaxespad=0., title="Legend", prop={'size': 8})
            # plt.legend(legend_labels, bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
            #            ncol=3, mode="expand", borderaxespad=0., title="Legend", prop={'size': 8})
            # plt.legend(legend_labels, ncol=len(legend_labels), loc="upper left", shadow=True, title="Legend",
            #            prop={'size': 8})
            plt.title(plot_titles[i] + " : Comparison b/w Actual and Generative Samples")
            plt.savefig(os.path.join(output_dir, str(each_point) + "_" + str(i) + "_" + "gs.png"))
            plt.close()


if __name__ == '__main__':
    # Step 1: Set up the input and the output directory.
    input_dir = "./results_worth_saving/generative_samples/"
    output_dir = "./results_worth_saving/generative_samples"

    # Step 2: Get the file names of the pickled files corresponding to each model and actual data. Note that there are
    # as many pickled files containing actual data as there are models. Though we'll be loading each of these, we'll be
    # using only one of those. The prefix 'nf' stands for 'NoFlow'. IT DOES NOT STAND FOR 'NORMALIZING FLOWS'.

    # Files containing actual and generative samples obtained from the 'NoFlow' model
    fname_nf_x = os.path.join(input_dir, "nf_x_for_generative_sampling_for_comparison.pkl")  # Actual data
    fname_nf_gs = os.path.join(input_dir, "nf_gs_samples_for_comparison.pkl")  # Generative samples

    fname_pf_x = os.path.join(input_dir, "pf_x_for_generative_sampling_for_comparison.pkl")  # Actual data
    fname_pf_gs = os.path.join(input_dir, "pf_gs_samples_for_comparison.pkl")  # Generative samples

    fname_rf_x = os.path.join(input_dir, "rf_x_for_generative_sampling_for_comparison.pkl")  # Actual data
    fname_rf_gs = os.path.join(input_dir, "rf_gs_samples_for_comparison.pkl")  # Generative samples

    # Step 3: Load pickled data
    nf_x = pickle.load(open(fname_nf_x, "rb"))
    nf_gs = pickle.load(open(fname_nf_gs, "rb"))

    pf_x = pickle.load(open(fname_pf_x, "rb"))
    pf_gs = pickle.load(open(fname_pf_gs, "rb"))

    rf_x = pickle.load(open(fname_pf_x, "rb"))
    rf_gs = pickle.load(open(fname_rf_gs, "rb"))

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

    pf_cos_gs = plots.helper_functions.sliceFrom3DTensor(pf_gs, 0)
    pf_sine_gs = plots.helper_functions.sliceFrom3DTensor(pf_gs, 1)
    pf_w_gs = plots.helper_functions.sliceFrom3DTensor(pf_gs, 2)  # Angular velocity omega
    pf_reward_gs = plots.helper_functions.sliceFrom3DTensor(pf_gs, 3)
    print pf_cos_gs.shape, pf_sine_gs.shape, pf_w_gs.shape, pf_reward_gs.shape

    rf_cos_gs = plots.helper_functions.sliceFrom3DTensor(rf_gs, 0)
    rf_sine_gs = plots.helper_functions.sliceFrom3DTensor(rf_gs, 1)
    rf_w_gs = plots.helper_functions.sliceFrom3DTensor(rf_gs, 2)  # Angular velocity omega
    rf_reward_gs = plots.helper_functions.sliceFrom3DTensor(rf_gs, 3)
    print rf_cos_gs.shape, rf_sine_gs.shape, rf_w_gs.shape, rf_reward_gs.shape

    # Prepare data for the plot function
    num_time_steps = nf_gs.shape[0]  # Get a list of time steps
    cut_off_time = 50
    num_features = nf_gs.shape[2] - 1
    flow_types = ["NF"]  # ["NF", "PF", "RF", "CPF"]
    plot_titles = ["Cosine", "Sine", "Angular Velocity", "Reward"]
    nf_actual_data = [nf_cos_actual, nf_sine_actual, nf_w_actual, nf_reward_actual]
    nf_gs_data = [nf_cos_gs, nf_sine_gs, nf_w_gs, nf_reward_gs]
    pf_gs_data = [pf_cos_gs, pf_sine_gs, pf_w_gs, pf_reward_gs]
    rf_gs_data = [rf_cos_gs, rf_sine_gs, rf_w_gs, rf_reward_gs]

    gs_data = [nf_gs_data, pf_gs_data, rf_gs_data]  # Add the data for gen signals from all the models in one list.

    # Step 4: Plot four separate plots for each of the input features. Each plot will have four curves.
    # One corresponding to the actual data until a certain time step t. After time step t, there will be curves
    # corresponding to the signals obtained from different models. The actual signal should appear in dotted after t.
    instances_to_plot = [0, 2, 4]
    plot_signals_and_generative_samples(num_time_steps, cut_off_time, num_features, nf_actual_data, gs_data, output_dir)




