import os
# from plots.helper_functions import *
import plots
import matplotlib.pyplot as plt


def compare_losses_across_models(output_dir, nof_losses, planar_losses):

    # Reconstruction Losses
    plt.figure(0)
    plt.plot(nof_losses[:, 0], nof_losses[:, 1])
    plt.plot(planar_losses[:, 0], planar_losses[:, 1])
    plt.title("Avg. Reconstruction Loss per Epoch")
    plt.ylabel("Avg. Reconstruction Loss")
    plt.xlabel("Epochs")
    legend_labels = ["STORN", "Planar"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "recons.png"))

    # KL Losses
    plt.figure(1)
    plt.plot(nof_losses[:, 0], nof_losses[:, 2])
    plt.plot(planar_losses[:, 0], planar_losses[:, 2])
    plt.title("Avg. KL Loss per Epoch")
    plt.ylabel("Avg. KL Loss")
    plt.xlabel("Epochs")
    legend_labels = ["STORN", "Planar"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "kl.png"))

    # Elbo
    plt.figure(2)
    plt.plot(nof_losses[:, 0], nof_losses[:, 3])
    plt.plot(planar_losses[:, 0], planar_losses[:, 3])
    plt.title("Avg. ELBO per Epoch")
    plt.ylabel("Avg. ELBO")
    plt.xlabel("Epochs")
    legend_labels = ["STORN", "Planar"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "elbo.png"))


# Entry point
if __name__ == '__main__':

    # Specify path to the log files obtained from different models
    no_flow_logfile_path = "/home/abhishek/Desktop/Junk/log_files/no_flow_logfile.log"
    planar_logfile_path = "/home/abhishek/Desktop/Junk/log_files/planar_logfile.log"

    output_dir = "/home/abhishek/Desktop/Junk/log_files/"  # Directory where the graphs will be saved.

    # Specify start and end epochs for the graph
    nepochs = 4000
    remove_outlier = True
    start_epoch = 1500
    end_epoch = 2500

    # Read log files
    nof_losses = plots.helper_functions.read_loss_logfile(no_flow_logfile_path)
    print "nof_losses shape:", nof_losses.shape
    planar_losses = plots.helper_functions.read_loss_logfile(planar_logfile_path)
    print "planar_losses shape:", planar_losses.shape

    nof_losses = nof_losses[start_epoch:end_epoch, :]
    planar_losses = planar_losses[start_epoch:end_epoch, :]
    if remove_outlier:
        # In the following line, as m increases, more outliers are removed.
        nof_losses = plots.helper_functions.remove_outliers(nof_losses, m=1)
        print "nof_losses sans outlier", nof_losses.shape
        planar_losses = plots.helper_functions.remove_outliers(planar_losses, m=100)
        print "planar_losses sans outlier", planar_losses.shape
    compare_losses_across_models(output_dir, nof_losses, planar_losses)



