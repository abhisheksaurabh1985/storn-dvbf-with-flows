import os
import numpy as np
# from plots.helper_functions import *
import plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from scipy import interpolate


def read_pdists_logfile(fname):
    """
    Read pdist.log to get the probability values. Returns a numpy array with number of epochs and all 5 values in the
    logfile.
    """
    phrases = ['Epoch:', 'Avg log_q0_z0:', 'Avg log_qk_zk:', 'Avg log_p_x_given_zk:',
               'Avg log_p_zk:', 'Avg sum_logdetj:']
    nepoch, log_q0_z0, log_qk_zk, log_p_x_given_zk, log_p_zk, sum_logdetj = [], [], [], [], [], []

    with open(fname) as f:
        for line in f:
            for x in phrases:
                line = line.replace(x, " ")
            words = line.split()

            if len(words) > 0:
                if words[0] == "Experiment":
                    continue
                nepoch.append(words[0])
                log_q0_z0.append(float(words[1]))
                log_qk_zk.append(float(words[2]))
                log_p_x_given_zk.append(float(words[3]))
                log_p_zk.append(float(words[4]))
                sum_logdetj.append(float(words[5]))

    num_epochs = np.expand_dims(np.asarray(nepoch), 1)
    avg_log_q0_z0 = np.expand_dims(np.asarray(log_q0_z0), 1)
    avg_log_qk_zk = np.expand_dims(np.asarray(log_qk_zk), 1)
    avg_log_p_x_given_zk = np.expand_dims(np.asarray(log_p_x_given_zk), 1)
    avg_log_p_zk = np.expand_dims(np.asarray(log_p_zk), 1)
    avg_sum_logdetj = np.expand_dims(np.asarray(sum_logdetj), 1)
    pdists_per_epoch = np.concatenate((num_epochs, avg_log_q0_z0, avg_log_qk_zk, avg_log_p_x_given_zk,
                                      avg_log_p_zk, avg_sum_logdetj), axis=1).astype(np.float32)
    return pdists_per_epoch


def compare_pdists_across_models(output_dir, pf_2_pd, pf_4_pd, pf_8_pd, rf_2_pd, rf_4_pd, rf_8_pd):
    # log_p(z_k)
    plt.figure("Comparison of log_p(z_k) across models")

    plt.plot(pf_2_pd[:, 0], pf_2_pd[:, 4])
    # plt.plot(pf_4_pd[:, 0], pf_4_pd[:, 4])
    plt.plot(pf_8_pd[:, 0], pf_8_pd[:, 4])

    plt.plot(rf_2_pd[:, 0], rf_2_pd[:, 4])
    # plt.plot(rf_4_pd[:, 0], rf_4_pd[:, 4])
    plt.plot(rf_8_pd[:, 0], rf_8_pd[:, 4])

    plt.title("Average $log\, p(z_k)$ per Epoch")
    plt.ylabel("$log\, p(z_k)$")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-8", "RF-2", "RF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "log_p_z_k.png"))

    plt.figure("Comparison of log_q_z0 across all Planar Flows")
    plt.plot(pf_2_pd[:, 0], pf_2_pd[:, 1])
    plt.plot(pf_4_pd[:, 0], pf_4_pd[:, 1])
    plt.plot(pf_8_pd[:, 0], pf_8_pd[:, 1])
    plt.title("Avg. $log\, q(z_0)$ per Epoch")
    plt.ylabel("$log\, q(z_0)$")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-4", "PF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "planar_" + "log_q_z_0.png"))

    plt.figure("Comparison of log_q_zk across all Planar Flows")
    plt.plot(pf_2_pd[:, 0], pf_2_pd[:, 2])
    plt.plot(pf_4_pd[:, 0], pf_4_pd[:, 2])
    plt.plot(pf_8_pd[:, 0], pf_8_pd[:, 2])
    plt.title("Avg. $log\, q(z_k)$ per Epoch")
    plt.ylabel("$log\, q(z_k)$")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-4", "PF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "planar_" + "log_q_z_k.png"))

    plt.figure("Comparison of sum_log_detj across all Planar Flows")
    plt.plot(pf_2_pd[:, 0], pf_2_pd[:, 5])
    plt.plot(pf_4_pd[:, 0], pf_4_pd[:, 5])
    plt.plot(pf_8_pd[:, 0], pf_8_pd[:, 5])
    plt.title("Avg. Sum Log Determinant of Jacobian per Epoch")
    plt.ylabel("Sum log det of jacobian")
    plt.xlabel("Epochs")
    legend_labels = ["PF-2", "PF-4", "PF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "planar_" + "sldj.png"))

    plt.figure("Comparison of log_q_z0 across all Radial Flows")
    plt.plot(rf_2_pd[:, 0], rf_2_pd[:, 1])
    plt.plot(rf_4_pd[:, 0], rf_4_pd[:, 1])
    plt.plot(rf_8_pd[:, 0], rf_8_pd[:, 1])
    plt.title("Avg. $log\, q(z_0)$ per Epoch")
    plt.ylabel("$log\, q(z_0)$")
    plt.xlabel("Epochs")
    legend_labels = ["RF-2", "RF-4", "RF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "radial_" + "log_q_z_0.png"))

    plt.figure("Comparison of log_q_zk across all Radial Flows")
    plt.plot(rf_2_pd[:, 0], rf_2_pd[:, 2])
    plt.plot(rf_4_pd[:, 0], rf_4_pd[:, 2])
    plt.plot(rf_8_pd[:, 0], rf_8_pd[:, 2])
    plt.title("Avg. $log\, q(z_0)$ per Epoch")
    plt.ylabel("$log\, q(z_0)$")
    plt.xlabel("Epochs")
    legend_labels = ["RF-2", "RF-4", "RF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "radial_" + "log_q_z_k.png"))

    plt.figure("Comparison of sum_log_detj across all Radial Flows")
    plt.plot(rf_2_pd[:, 0], rf_2_pd[:, 5])
    plt.plot(rf_4_pd[:, 0], rf_4_pd[:, 5])
    plt.plot(rf_8_pd[:, 0], rf_8_pd[:, 5])
    plt.title("Avg. Sum Log Determinant of Jacobian per Epoch")
    plt.ylabel("Sum log det of jacobian")
    plt.xlabel("Epochs")
    legend_labels = ["RF-2", "RF-4", "RF-8"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    plt.savefig(os.path.join(output_dir, "radial_" + "sldj.png"))


# Entry point
if __name__ == '__main__':
    # Specify path to the log files obtained from different models. Flow length k=8 is used here for planar and radial.
    # no_flow_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/" \
    #                        "results_worth_saving/raw_worthy_data/best_results/no_flow_2017_08_26_22_44_06/pdists.log"

    pf_2_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/planar/2017_08_29_11_30_56/pdists.log"
    pf_4_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/planar/2017_08_29_13_51_44/pdists.log"
    pf_8_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/planar/2017_08_27_11_47_32/pdists.log"

    rf_2_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/radial/2017_08_30_23_27_26/pdists.log"
    rf_4_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/radial/2017_08_31_10_36_48/pdists.log"
    rf_8_pdists_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                        "flow_length_comparison/radial/2017_08_28_10_22_59/pdists.log"

    output_dir = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/" \
                 "results_worth_saving/pdist_comparison_plots/"  # Directory where the graphs will be saved.

    # Specify start and end epochs for the graph
    nepochs = 10000
    start_epoch = 4000
    end_epoch = 90000

    # Read log files
    pf_2_pdists = read_pdists_logfile(pf_2_pdists_logfile_path)
    pf_4_pdists = read_pdists_logfile(pf_4_pdists_logfile_path)
    pf_8_pdists = read_pdists_logfile(pf_8_pdists_logfile_path)

    rf_2_pdists = read_pdists_logfile(rf_2_pdists_logfile_path)
    rf_4_pdists = read_pdists_logfile(rf_4_pdists_logfile_path)
    rf_8_pdists = read_pdists_logfile(rf_8_pdists_logfile_path)

    pf_2_pdists = pf_2_pdists[start_epoch:end_epoch, :]
    pf_4_pdists = pf_4_pdists[start_epoch:end_epoch, :]
    pf_8_pdists = pf_8_pdists[start_epoch:end_epoch, :]

    rf_2_pdists = rf_2_pdists[start_epoch:end_epoch, :]
    rf_4_pdists = rf_4_pdists[start_epoch:end_epoch, :]
    rf_8_pdists = rf_8_pdists[start_epoch:end_epoch, :]

    compare_pdists_across_models(output_dir, pf_2_pdists, pf_4_pdists, pf_8_pdists,
                                 rf_2_pdists, rf_4_pdists, rf_8_pdists)
