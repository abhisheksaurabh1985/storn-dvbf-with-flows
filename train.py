import os
import time
import numpy as np
from matplotlib import pyplot as plt

import plots
from data_source import dataset_utils


def train_nf(sess, loss_op, loss_summary, prob_dists, solver, nepochs, n_samples, batch_size,
             display_step, _X, data, summary=None, file_writer=None, flow_type=None, output_dir=None):

    avg_recons_loss, avg_kl_loss, avg_elbo_loss = [], [], []
    avg_log_q0_z0, avg_log_qk_zk, avg_log_p_x_given_zk, avg_log_p_zk, avg_sum_logdetj = [], [], [], [], []
    start_time = time.time()
    print "########## Training Starts ##########"

    for epoch in range(nepochs):
        avg_recons_loss_per_epoch = avg_kl_loss_per_epoch = avg_elbo_loss_per_epoch = 0
        avg_log_q0_z0_per_epoch = avg_log_qk_zk_per_epoch = avg_log_p_x_given_zk_per_epoch = \
            avg_log_p_zk_per_epoch = avg_sum_logdetj_per_epoch = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data.train.next_batch(batch_size)
            # batch_xs = dataset_utils.normalize_data(batch_xs)  # Use z-score normalization
            _, cost, res_summary, probability_distributions = sess.run([solver, loss_op, summary, prob_dists],
                                                                       feed_dict={_X: batch_xs})
            recons_loss = cost[0]
            kl_loss = cost[1]
            elbo_loss = cost[2]

            log_q0_z0 = probability_distributions[0]
            log_qk_zk = probability_distributions[1]
            log_p_x_given_zk = probability_distributions[2]
            log_p_zk = probability_distributions[3]
            sum_logdetj = probability_distributions[4]

            file_writer.add_summary(res_summary, epoch * total_batch + i)

            # Average losses per epoch
            avg_recons_loss_per_epoch += (recons_loss / n_samples) * batch_size
            avg_kl_loss_per_epoch += (kl_loss / n_samples) * batch_size
            avg_elbo_loss_per_epoch += (elbo_loss / n_samples) * batch_size

            # Average probability density per epoch
            avg_log_q0_z0_per_epoch += (log_q0_z0 / n_samples) * batch_size
            avg_log_qk_zk_per_epoch += (log_qk_zk / n_samples) * batch_size
            avg_log_p_x_given_zk_per_epoch += (log_p_x_given_zk / n_samples) * batch_size
            avg_log_p_zk_per_epoch += (log_p_zk / n_samples) * batch_size
            avg_sum_logdetj_per_epoch += (sum_logdetj / n_samples) * batch_size

        # Average loss
        avg_recons_loss.append(avg_recons_loss_per_epoch)
        avg_kl_loss.append(avg_kl_loss_per_epoch)
        avg_elbo_loss.append(avg_elbo_loss_per_epoch)

        # Average probabilities
        avg_log_q0_z0.append(avg_log_q0_z0_per_epoch)
        avg_log_qk_zk.append(avg_log_qk_zk_per_epoch)
        avg_log_p_x_given_zk.append(avg_log_p_x_given_zk_per_epoch)
        avg_log_p_zk.append(avg_log_p_zk_per_epoch)
        avg_sum_logdetj.append(avg_sum_logdetj_per_epoch)

        if epoch % display_step == 0:
            print "shape avg_recons_loss[epoch]", avg_recons_loss[epoch].shape
            print "shape avg_kl_loss[epoch]", avg_kl_loss[epoch].shape
            print "shape avg_recons_loss[epoch]", avg_elbo_loss[epoch].shape
            print "type avg_recons_loss[epoch]", type(float(avg_recons_loss[epoch]))
            print "type avg_kl_loss[epoch]", type(float(avg_kl_loss[epoch]))
            print "type avg_elbo_loss[epoch]", type(float(avg_elbo_loss[epoch]))
            line_losses = "Epoch: %i \t Average recons loss: %0.4f \t Average kl loss: %0.4f \t " \
                          "Average elbo loss: %0.4f" % \
                (epoch,
                    float(np.round(avg_recons_loss[epoch], 4)),
                    float(np.round(avg_kl_loss[epoch], 4)),
                    float(np.round(avg_elbo_loss[epoch], 4)))
            print line_losses
            with open(os.path.join(output_dir, "logfile.log"), "a") as f:
                f.write(line_losses + "\n")

            line_pdists = "Epoch: %i \t Avg log_q0_z0: %0.4f \t Avg log_qk_zk: %0.4f \t " \
                          "Avg log_p_x_given_zk: %0.4f \t Avg log_p_zk: %0.4f \t Avg sum_logdetj: %0.4f" % \
                (epoch,
                    float(round(avg_log_q0_z0[epoch], 4)),
                    float(round(avg_log_qk_zk[epoch], 4)),
                    float(round(avg_log_p_x_given_zk[epoch], 4)),
                    float(round(avg_log_p_zk[epoch], 4)),
                    float(round(avg_sum_logdetj[epoch], 4)))
            print line_pdists
            with open(os.path.join(output_dir, "pdists.log"), "a") as f:
                f.write(line_pdists + "\n")

    print("--- %s seconds ---" % (time.time() - start_time))

    # Plot losses per epoch
    plots.plots.plot_losses_for_nf(nepochs, avg_recons_loss, avg_kl_loss, avg_elbo_loss, flow_type, output_dir)
    # Plot probability densities
    # if flow_type == "Planar" or flow_type == "ConvolutionPlanar" or flow_type == "Radial":
    plots.plots.plot_probability_densities(nepochs, avg_log_q0_z0, avg_log_qk_zk, avg_log_p_x_given_zk, avg_log_p_zk,
                                           avg_sum_logdetj, flow_type, output_dir)
    return avg_elbo_loss


def train(sess, loss_op, solver, nepochs, n_samples, batch_size,
          display_step, _X, data, summary=None, file_writer=None, flow_type=None, output_dir=None):
    """
    Training for vanilla STORN/ DVBF
    """
    avg_vae_loss = []
    start_time = time.time()
    print "###### Training starts ######"

    for epoch in range(nepochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data.train.next_batch(batch_size)
            _, cost, res_summary = sess.run([solver, loss_op, summary], feed_dict={_X: batch_xs})
            file_writer.add_summary(res_summary, epoch * total_batch + i)
            avg_cost += (cost / n_samples) * batch_size
        avg_vae_loss.append(avg_cost)
        if epoch % display_step == 0:
            line = "Epoch: %i \t Average cost: %0.9f" % (epoch, avg_vae_loss[epoch])
            print line
            with open(os.path.join(output_dir, "logfile.log"), "a") as f:
                f.write(line + "\n")
    print("--- %s seconds ---" % (time.time() - start_time))
    # Plot training loss per epoch
    plt.plot(range(nepochs), avg_vae_loss)
    plt.title("Average loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Average Loss")
    plt.savefig(os.path.join(output_dir, flow_type + "_" + "losses.png"))
    # plt.show()
    return avg_vae_loss


