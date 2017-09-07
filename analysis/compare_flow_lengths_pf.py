import os
# from plots.helper_functions import *
import plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from scipy import interpolate


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    http://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def compare_losses_across_models_with_sg_filter(output_dir, pf_2_losses, pf_4_losses, pf_8_losses):
    """
    This function uses the Savitzky- Golay filter. Losses are very noisy.
    :param output_dir:
    :param nof_losses:
    :param planar_losses:
    :return:
    """
    # Reconstruction Losses
    plt.figure(0)

    pf_2_losses[:, 1] = savitzky_golay(pf_2_losses[:, 1], 181, 3)  # window size 51, polynomial order 3
    pf_4_losses[:, 1] = savitzky_golay(pf_4_losses[:, 1], 181, 3)  # window size 51, polynomial order 3
    pf_8_losses[:, 1] = savitzky_golay(pf_8_losses[:, 1], 181, 3)  # window size 51, polynomial order 3

    plt.plot(pf_2_losses[:, 0], pf_2_losses[:, 1])
    plt.plot(pf_4_losses[:, 0], pf_4_losses[:, 1])
    plt.plot(pf_8_losses[:, 0], pf_8_losses[:, 1])

    plt.title("Avg. Reconstruction Loss per Epoch")
    plt.ylabel("Avg. Reconstruction Loss")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-4", "PF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "recons.png"))

    # KL Losses
    plt.figure(1)

    pf_2_losses[:, 2] = savitzky_golay(pf_2_losses[:, 2], 181, 3)  # window size 51, polynomial order 3
    pf_4_losses[:, 2] = savitzky_golay(pf_4_losses[:, 2], 181, 3)  # window size 51, polynomial order 3
    pf_8_losses[:, 2] = savitzky_golay(pf_8_losses[:, 2], 181, 3)  # window size 51, polynomial order 3

    plt.plot(pf_2_losses[:, 0], pf_2_losses[:, 2])
    plt.plot(pf_4_losses[:, 0], pf_4_losses[:, 2])
    plt.plot(pf_8_losses[:, 0], pf_8_losses[:, 2])
    plt.title("Avg. KL Loss per Epoch")
    plt.ylabel("Avg. KL Loss")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-4", "PF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "kl.png"))

    # Elbo
    plt.figure(2)

    pf_2_losses[:, 3] = savitzky_golay(pf_2_losses[:, 3], 181, 3)  # window size 51, polynomial order 3
    pf_4_losses[:, 3] = savitzky_golay(pf_4_losses[:, 3], 181, 3)  # window size 51, polynomial order 3
    pf_8_losses[:, 3] = savitzky_golay(pf_8_losses[:, 3], 181, 3)  # window size 51, polynomial order 3

    plt.plot(pf_2_losses[:, 0], pf_2_losses[:, 3])
    plt.plot(pf_4_losses[:, 0], pf_4_losses[:, 3])
    plt.plot(pf_8_losses[:, 0], pf_8_losses[:, 3])
    plt.title("Avg. ELBO per Epoch")
    plt.ylabel("Avg. ELBO")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-4", "PF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "elbo.png"))


def compare_losses_across_models(output_dir, pf_2_losses, pf_4_losses, pf_8_losses):
    """
    This one doesn't apply any smoothing.
    :param output_dir:
    :param nof_losses:
    :param planar_losses:
    :return:
    """
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
    # no_flow_logfile_path = "/home/abhishek/Desktop/Junk/log_files/no_flow_logfile.log"
    # planar_logfile_path = "/home/abhishek/Desktop/Junk/log_files/planar_logfile.log"
    pf_2_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/planar/2017_08_29_11_30_56/logfile.log"
    pf_4_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/planar/2017_08_29_13_51_44/logfile.log"
    pf_8_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/planar/2017_08_27_11_47_32/logfile.log"

    # output_dir = "/home/abhishek/Desktop/Junk/log_files/"  # Directory where the graphs will be saved.
    output_dir = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                 "flow_length_comparison/"

    # Specify start and end epochs for the graph
    nepochs = 10000
    remove_outlier = False
    start_epoch = 5000
    end_epoch = 9000

    # Read log files
    pf_2_losses = plots.helper_functions.read_loss_logfile(pf_2_logfile_path)
    print "pf_2_losses shape:", pf_2_losses.shape
    pf_4_losses = plots.helper_functions.read_loss_logfile(pf_4_logfile_path)
    print "pf_4_losses shape:", pf_4_losses.shape
    pf_8_losses = plots.helper_functions.read_loss_logfile(pf_8_logfile_path)
    print "pf_8_losses shape:", pf_8_losses.shape
    # nof_losses = plots.helper_functions.read_loss_logfile(no_flow_logfile_path)
    # print "nof_losses shape:", nof_losses.shape
    # planar_losses = plots.helper_functions.read_loss_logfile(planar_logfile_path)
    # print "planar_losses shape:", planar_losses.shape

    pf_2_losses = pf_2_losses[start_epoch:end_epoch, :]
    pf_4_losses = pf_4_losses[start_epoch:end_epoch, :]
    pf_8_losses = pf_8_losses[start_epoch:end_epoch, :]

    # nof_losses = nof_losses[start_epoch:end_epoch, :]
    # planar_losses = planar_losses[start_epoch:end_epoch, :]
    if remove_outlier:
        # In the following line, as m increases, more outliers are removed.
        pf_2_losses = plots.helper_functions.remove_outliers(pf_2_losses, m=1)
        print "pf_2_losses sans outlier", pf_2_losses.shape
        pf_4_losses = plots.helper_functions.remove_outliers(pf_4_losses, m=1)
        print "pf_4_losses sans outlier", pf_4_losses.shape
        pf_8_losses = plots.helper_functions.remove_outliers(pf_8_losses, m=1)
        print "pf_8_losses sans outlier", pf_8_losses.shape
        compare_losses_across_models_with_sg_filter(output_dir, pf_2_losses, pf_4_losses, pf_8_losses)
    else:
        compare_losses_across_models_with_sg_filter(output_dir, pf_2_losses, pf_4_losses, pf_8_losses)


