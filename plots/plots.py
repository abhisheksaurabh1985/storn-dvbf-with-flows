import os
from matplotlib import pyplot as plt


def line_plot_2d(x, y):
    plt.plot(x, y)
    return


def plot_signals_and_reconstructions(time_steps, actual, recons, flow_type=None, output_dir=None,
                                     points_to_plot=[0, 2, 4, 6, 8, 10]):
    for each_point in points_to_plot:
        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(2, 2)
        f.set_size_inches(10, 6)
        f.suptitle("Actual and Reconstructed Signals:" + "Instance " + str(each_point) + " in batch\n\n", fontsize="x-large")

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
        f.savefig(os.path.join(output_dir, str(each_point) + "_" + flow_type + "_" + "recons_signals.png"))
        # plt.show()


def plot_losses_for_nf(nepochs, avg_recons_loss, avg_kl_loss, avg_elbo_loss, flow_type=None, output_dir=None):
    """
    Plots reconstruction, kl and elbo loss for STORN/ DVBF with flow.
    """
    # Three subplots with a shared x-axis
    f, axarr = plt.subplots(3, sharex=True)
    f.suptitle("Average Reconstruction, KL and ELBO loss", fontsize="x-large")

    range_epochs = range(nepochs)

    axarr[0].plot(range_epochs, avg_recons_loss)
    axarr[0].set_title("Average Reconstruction Loss")
    axarr[0].set_xlabel("Number of epochs")
    axarr[0].set_ylabel("Average Reconstruction Loss")

    axarr[1].plot(range_epochs, avg_kl_loss)
    axarr[1].set_title("Average KL Loss")
    axarr[1].set_xlabel("Number of epochs")
    axarr[1].set_ylabel("Average KL Loss")

    axarr[2].plot(range_epochs, avg_elbo_loss)
    axarr[2].set_title("Average ELBO Loss")
    axarr[2].set_xlabel("Number of epochs")
    axarr[2].set_ylabel("Average Reconstruction Loss")

    f.tight_layout()
    f.savefig(os.path.join(output_dir, flow_type + "_" + "losses.png"))
    # plt.show()
    # plt.clf()


def test_plot_actual_signals(time_steps, actual, points_to_plot=[0, 2, 4, 6, 8, 10]):
    for each_point in points_to_plot:
        # Four axes, returned as a 2-d array
        f, axarr = plt.subplots(2, 2)
        f.set_size_inches(10, 6)
        f.suptitle("Actual Signals:" + "Instance " + str(each_point) + " in batch\n\n", fontsize="x-large")

        axarr[0, 0].plot(time_steps, actual[0][:, each_point])
        # axarr[0, 0].plot(time_steps, recons[0][:, each_point])
        axarr[0, 0].set_title('Cosine')

        axarr[0, 1].plot(time_steps, actual[1][:, each_point])
        # axarr[0, 1].plot(time_steps, recons[1][:, each_point])
        axarr[0, 1].set_title('Sine')

        axarr[1, 0].plot(time_steps, actual[2][:, each_point])
        # axarr[1, 0].plot(time_steps, recons[2][:, each_point])
        axarr[1, 0].set_title('Velocity')

        axarr[1, 1].plot(time_steps, actual[3][:, each_point])
        # axarr[1, 1].plot(time_steps, recons[3][:, each_point])
        axarr[1, 1].set_title('Reward')

        legend_labels = ["Actual"]
        plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
        f.tight_layout()
        # f.savefig(os.path.join(output_dir, str(each_point) + "_" + flow_type + "_" + "recons_signals.png"))
        plt.show()
