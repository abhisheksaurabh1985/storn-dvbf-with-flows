import numpy as np

import plots

# Specify path to the log files obtained from different models. Flow length k=8 is used here for planar and radial.
no_flow_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                       "raw_worthy_data/best_results/no_flow_2017_08_26_22_44_06/logfile.log"

# Planar flow
pf_2_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                    "flow_length_comparison/planar/2017_08_29_11_30_56/logfile.log"
pf_4_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                    "flow_length_comparison/planar/2017_08_29_13_51_44/logfile.log"
pf_8_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                    "flow_length_comparison/planar/2017_08_27_11_47_32/logfile.log"

rf_2_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                    "flow_length_comparison/radial/2017_08_30_23_27_26/logfile.log"
rf_4_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                    "flow_length_comparison/radial/2017_08_31_10_36_48/logfile.log"
rf_8_logfile_path = "/home/abhishek/Dropbox/MAI/Internship-Thesis/TUM-Smagt/Thesis/results_worth_saving/" \
                    "flow_length_comparison/radial/2017_08_28_10_22_59/logfile.log"

# Read log files
nof_losses = plots.helper_functions.read_loss_logfile(no_flow_logfile_path)
avg_nof_losses = np.mean(nof_losses, axis=0)
print "nof_losses shape:", nof_losses.shape
print "avg_nof_losses shape:", avg_nof_losses.shape
print "\nAverage NOF losses:\n"
print "Epochs", np.round(avg_nof_losses[0], 2)
print "ELBO", np.round(avg_nof_losses[1], 2)
print "KL", np.round(avg_nof_losses[2], 2)
print "RL", np.round(avg_nof_losses[3], 2)


pf_2_losses = plots.helper_functions.read_loss_logfile(pf_2_logfile_path)
avg_pf_2_losses = np.mean(pf_2_losses, axis=0)
print "pf_2_losses shape:", pf_2_losses.shape
print "avg_pf_2_losses shape:", avg_pf_2_losses.shape
print "\nAverage PF-2 losses:\n"
print "Epochs", np.round(avg_pf_2_losses[0], 2)
print "ELBO", np.round(avg_pf_2_losses[1], 2)
print "KL", np.round(avg_pf_2_losses[2], 2)
print "RL", np.round(avg_pf_2_losses[3], 2)

pf_4_losses = plots.helper_functions.read_loss_logfile(pf_4_logfile_path)
avg_pf_4_losses = np.mean(pf_4_losses, axis=0)
print "pf_4_losses shape:", pf_4_losses.shape
print "avg_pf_4_losses shape:", avg_pf_4_losses.shape
print "\nAverage PF-4 losses:\n"
print "Epochs", np.round(avg_pf_4_losses[0], 2)
print "ELBO", np.round(avg_pf_4_losses[1], 2)
print "KL", np.round(avg_pf_4_losses[2], 2)
print "RL", np.round(avg_pf_4_losses[3], 2)


pf_8_losses = plots.helper_functions.read_loss_logfile(pf_8_logfile_path)
avg_pf_8_losses = np.mean(pf_8_losses, axis=0)
print "pf_8_losses shape:", pf_8_losses.shape
print "avg_pf_8_losses shape:", avg_pf_8_losses.shape
print "\nAverage PF-8 losses:\n"
print "Epochs", np.round(avg_pf_8_losses[0], 2)
print "ELBO", np.round(avg_pf_8_losses[1], 2)
print "KL", np.round(avg_pf_8_losses[2], 2)
print "RL", np.round(avg_pf_8_losses[3], 2)

rf_2_losses = plots.helper_functions.read_loss_logfile(rf_2_logfile_path)
avg_rf_2_losses = np.mean(rf_2_losses, axis=0)
print "rf_2_losses shape:", rf_2_losses.shape
print "avg_rf_2_losses shape:", avg_rf_2_losses.shape
print "\nAverage RF-2 losses:\n"
print "Epochs", np.round(avg_rf_2_losses[0], 2)
print "ELBO", np.round(avg_rf_2_losses[1], 2)
print "KL", np.round(avg_rf_2_losses[2], 2)
print "RL", np.round(avg_rf_2_losses[3], 2)


rf_4_losses = plots.helper_functions.read_loss_logfile(rf_4_logfile_path)
avg_rf_4_losses = np.mean(rf_4_losses, axis=0)
print "rf_4_losses shape:", rf_4_losses.shape
print "avg_rf_4_losses shape:", avg_rf_4_losses.shape
print "\nAverage RF-4 losses:\n"
print "Epochs", np.round(avg_rf_4_losses[0], 2)
print "ELBO", np.round(avg_rf_4_losses[1], 2)
print "KL", np.round(avg_rf_4_losses[2], 2)
print "RL", np.round(avg_rf_4_losses[3], 2)


rf_8_losses = plots.helper_functions.read_loss_logfile(rf_8_logfile_path)
avg_rf_8_losses = np.mean(rf_8_losses, axis=0)
print "rf_8_losses shape:", rf_8_losses.shape
print "avg_rf_8_losses shape:", avg_rf_8_losses.shape
print "\nAverage RF-8 losses:\n"
print "Epochs", np.round(avg_rf_8_losses[0], 2)
print "ELBO", np.round(avg_rf_8_losses[1], 2)
print "KL", np.round(avg_rf_8_losses[2], 2)
print "RL", np.round(avg_rf_8_losses[3], 2)