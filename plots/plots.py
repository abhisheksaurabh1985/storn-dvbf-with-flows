import os
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-deep')
from moviepy.editor import ImageSequenceClip


def line_plot_2d(x, y):
    plt.plot(x, y)
    return


def distribution_signals(data, dir, flow_type, signal, points_to_plot=[0, 2]):
    for each_point in points_to_plot:
        f, axarr = plt.subplots(2, 2)
        f.set_size_inches(10, 6)
        f.suptitle("Distribution of values:" + "Instance " + str(each_point) + " in batch\n\n",
                   fontsize="x-large")

        axarr[0, 0].hist(data[:, each_point, 0].reshape((-1,)))
        axarr[0, 0].set_title('Cosine')

        axarr[0, 1].hist(data[:, each_point, 1].reshape((-1,)))
        axarr[0, 1].set_title('Sine')

        axarr[1, 0].hist(data[:, each_point, 2].reshape((-1,)))
        axarr[1, 0].set_title('Angular Velocity')

        axarr[1, 1].hist(data[:, each_point, 3].reshape((-1,)))
        axarr[1, 1].set_title('Reward')

        legend_labels = ["Data Distribution"]
        plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
        f.tight_layout()
        f.savefig(os.path.join(dir, str(each_point) + "_" + str.lower(signal) + "_" + "dist_signals.png"))


def plot_generative_samples(time_steps, gs, flow_type=None, output_dir=None,
                            points_to_plot=[0, 2]):
    for each_point in points_to_plot:
        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(2, 2)
        f.set_size_inches(10, 6)
        f.suptitle("Generative Signals:" + "Instance " + str(each_point) + " in batch\n\n",
                   fontsize="x-large")

        axarr[0, 0].plot(time_steps, gs[0][:, each_point])
        axarr[0, 0].set_title('Cosine')

        axarr[0, 1].plot(time_steps, gs[1][:, each_point])
        axarr[0, 1].set_title('Sine')

        axarr[1, 0].plot(time_steps, gs[2][:, each_point])
        axarr[1, 0].set_title('Velocity')

        axarr[1, 1].plot(time_steps, gs[3][:, each_point])
        axarr[1, 1].set_title('Reward')

        legend_labels = ["Generative sample"]
        plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
        f.tight_layout()
        f.savefig(os.path.join(output_dir, "gs/", str(each_point) + "_" + str.lower(flow_type) + "_" + "gs_signals.png"))


def plot_signals_and_reconstructions(time_steps, actual, recons, flow_type=None, output_dir=None,
                                     points_to_plot=[0, 2, 4, 6, 8, 10]):
    for each_point in points_to_plot:
        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(2, 2)
        f.set_size_inches(10, 6)
        f.suptitle("Actual and Reconstructed Signals:" + "Instance " + str(each_point) + " in batch\n\n",
                   fontsize="x-large")

        axarr[0, 0].plot(time_steps, actual[0][:, each_point])
        axarr[0, 0].plot(time_steps, recons[0][:, each_point])
        axarr[0, 0].set_title('Cosine')

        axarr[0, 1].plot(time_steps, actual[1][:, each_point])
        axarr[0, 1].plot(time_steps, recons[1][:, each_point])
        axarr[0, 1].set_title('Sine')

        axarr[1, 0].plot(time_steps, actual[2][:, each_point])
        axarr[1, 0].plot(time_steps, recons[2][:, each_point])
        axarr[1, 0].set_title('Velocity')

        axarr[1, 1].plot(time_steps, actual[3][:, each_point])
        axarr[1, 1].plot(time_steps, recons[3][:, each_point])
        axarr[1, 1].set_title('Reward')

        legend_labels = ["Actual", "Reconstruction"]
        plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
        f.tight_layout()
        f.savefig(os.path.join(output_dir, str(each_point) + "_" + str.lower(flow_type) + "_" + "recons_signals.png"))


def plot_losses_for_nf(nepochs, avg_recons_loss, avg_kl_loss, avg_elbo_loss, flow_type=None, output_dir=None):
    """
    Plots reconstruction, kl and elbo loss for STORN/ DVBF with flow. Separate plot for each loss.
    """
    # Three separate plots
    range_epochs = range(nepochs)

    plt.figure("Average Reconstruction Loss per Epoch")
    plt.plot(range_epochs, avg_recons_loss)
    plt.title("Average Reconstruction Loss per Epoch")
    plt.ylabel("Average Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "recons.png"))

    plt.figure("Average KL Loss per Epoch")
    plt.plot(range_epochs, avg_kl_loss)
    plt.title("Average KL Loss per Epoch")
    plt.ylabel("Average KL Loss")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "kl.png"))

    plt.figure("Average ELBO per Epoch")
    plt.plot(range_epochs, avg_elbo_loss)
    plt.title("Average ELBO per Epoch")
    plt.ylabel("Average ELBO")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "elbo.png"))


def plot_probability_densities(nepochs, log_q0_z0, log_qk_zk, log_p_x_given_zk, log_p_zk, avg_sum_logdetj,
                               flow_type=None, output_dir=None):
    # Three separate plots
    range_epochs = range(nepochs)

    plt.figure("Average $log\, q_0z_0$ per epoch")
    plt.plot([int(elm) for elm in range_epochs], [float(round(elem, 2)) for elem in log_q0_z0])
    # plt.plot(range_epochs, log_q0_z0)
    plt.title("Average $log\, q_0z_0$ per epoch")
    plt.ylabel("Average $log q_0z_0$")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "log_q0_z0.png"))

    plt.figure("Average $log\, q_kz_k$ per epoch")
    plt.plot(range_epochs, log_qk_zk)
    plt.title("Average $log\, q_kz_k$ per epoch")
    plt.ylabel("Average $log q_kz_k$")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "log_qk_zk.png"))

    plt.figure("Average $log\, p(x|z_k)$ per Epoch")
    plt.plot(range_epochs, log_p_x_given_zk)
    plt.title("Average $log\, p(x|z_k)$ per Epoch")
    plt.ylabel("Average $log p(x|z_k)$")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "log_p_x_given_zk.png"))

    plt.figure("Average $log\, p(z_k)$ per Epoch")
    plt.plot(range_epochs, log_p_zk)
    plt.title("Average $log\, p(z_k)$ per Epoch")
    plt.ylabel("Average $log\, p(z_k)$")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "log_p_zk.png"))

    plt.figure("Average Sum Log Determinant of Jacobian per Epoch")
    plt.plot(range_epochs, avg_sum_logdetj)
    plt.title("Average Sum Log Determinant of Jacobian per Epoch")
    plt.ylabel("Average avg_sum_logdetj")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(output_dir, str.lower(flow_type) + "_" + "sum_logdetj.png"))

# def test_plot_actual_signals(time_steps, actual, points_to_plot=[0, 2, 4, 6, 8, 10]):
#     """
#     Plot for actual signals. Checked with Max. These are correct. Probably will not be used in future.
#     :param time_steps:
#     :param actual:
#     :param points_to_plot:
#     :return:
#     """
#     for each_point in points_to_plot:
#         # Four axes, returned as a 2-d array
#         f, axarr = plt.subplots(2, 2)
#         f.set_size_inches(10, 6)
#         f.suptitle("Actual Signals:" + "Instance " + str(each_point) + " in batch\n\n", fontsize="x-large")
#
#         axarr[0, 0].plot(time_steps, actual[0][:, each_point])
#         # axarr[0, 0].plot(time_steps, recons[0][:, each_point])
#         axarr[0, 0].set_title('Cosine')
#
#         axarr[0, 1].plot(time_steps, actual[1][:, each_point])
#         # axarr[0, 1].plot(time_steps, recons[1][:, each_point])
#         axarr[0, 1].set_title('Sine')
#
#         axarr[1, 0].plot(time_steps, actual[2][:, each_point])
#         # axarr[1, 0].plot(time_steps, recons[2][:, each_point])
#         axarr[1, 0].set_title('Velocity')
#
#         axarr[1, 1].plot(time_steps, actual[3][:, each_point])
#         # axarr[1, 1].plot(time_steps, recons[3][:, each_point])
#         axarr[1, 1].set_title('Reward')
#
#         legend_labels = ["Actual"]
#         plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
#         f.tight_layout()
#         # f.savefig(os.path.join(output_dir, str(each_point) + "_" + flow_type + "_" + "recons_signals.png"))
#         plt.show()

# def plot_losses_for_nf(nepochs, avg_recons_loss, avg_kl_loss, avg_elbo_loss, flow_type=None, output_dir=None):
#     """
#     Plots reconstruction, kl and elbo loss for STORN/ DVBF with flow.
#     """
#     # Three subplots with a shared x-axis
#     f, axarr = plt.subplots(3, sharex=True)
#     f.suptitle("Average Reconstruction, KL and ELBO loss", fontsize="x-large")
#
#     range_epochs = range(nepochs)
#
#     axarr[0].plot(range_epochs, avg_recons_loss)
#     axarr[0].set_title("Average Reconstruction Loss")
#     axarr[0].set_xlabel("Number of epochs")
#     axarr[0].set_ylabel("Average Reconstruction Loss")
#
#     axarr[1].plot(range_epochs, avg_kl_loss)
#     axarr[1].set_title("Average KL Loss")
#     axarr[1].set_xlabel("Number of epochs")
#     axarr[1].set_ylabel("Average KL Loss")
#
#     axarr[2].plot(range_epochs, avg_elbo_loss)
#     axarr[2].set_title("Average ELBO Loss")
#     axarr[2].set_xlabel("Number of epochs")
#     axarr[2].set_ylabel("Average Reconstruction Loss")
#
#     f.tight_layout()
#     f.savefig(os.path.join(output_dir, flow_type + "_" + "losses.png"))
#     # plt.show()
#     # plt.clf()
#
