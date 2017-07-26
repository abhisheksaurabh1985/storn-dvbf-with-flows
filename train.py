import time
import plots
from matplotlib import pyplot as plt


def train_nf(sess, loss_op, solver, nepochs, n_samples, batch_size,
             display_step, _X, data, summary=None, file_writer=None):
    """
    Training for planar normalizing flows
    """
    avg_recons_loss, avg_kl_loss, avg_elbo_loss = [], [], []
    start_time = time.time()
    print "###### Training starts ######"
    for epoch in range(nepochs):
        avg_recons_loss_per_epoch = avg_kl_loss_per_epoch = avg_elbo_loss_per_epoch = 0
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data.train.next_batch(batch_size)
            _, cost, res_summary = sess.run([solver, loss_op, summary], feed_dict={_X: batch_xs})
            recons_loss = cost[0]
            kl_loss = cost[1]
            elbo_loss = cost[2]
            file_writer.add_summary(res_summary, epoch * total_batch + i)

            # Average losses per epoch
            avg_recons_loss_per_epoch += (recons_loss / n_samples) * batch_size
            avg_kl_loss_per_epoch += (kl_loss / n_samples) * batch_size
            avg_elbo_loss_per_epoch += (elbo_loss / n_samples) * batch_size

        # Average loss
        avg_recons_loss.append(avg_recons_loss_per_epoch)
        avg_kl_loss.append(avg_kl_loss_per_epoch)
        avg_elbo_loss.append(avg_elbo_loss_per_epoch)

        if epoch % display_step == 0:
            print "shape avg_recons_loss[epoch]", avg_recons_loss[epoch].shape
            print "shape avg_kl_loss[epoch]", avg_kl_loss[epoch].shape
            print "shape avg_recons_loss[epoch]", avg_elbo_loss[epoch].shape
            print "type avg_recons_loss[epoch]", type(float(avg_recons_loss[epoch]))
            print "type avg_kl_loss[epoch]", type(float(avg_kl_loss[epoch]))
            print "type avg_elbo_loss[epoch]", type(float(avg_elbo_loss[epoch]))
            line = "Epoch: %i \t Average recons loss: %0.9f \t Average kl loss: %0.9f \t Average elbo loss: %0.9f" % \
                   (epoch,
                    float(avg_recons_loss[epoch]),
                    float(avg_kl_loss[epoch]),
                    float(avg_elbo_loss[epoch]))
            print line
            with open("./output/logfile.log", "a") as f:
                f.write(line + "\n")
    print("--- %s seconds ---" % (time.time() - start_time))

    print "len avg_recons_loss:", len(avg_recons_loss)
    print "len avg_kl_loss:", len(avg_kl_loss)
    print "len avg_elbo_loss:", len(avg_elbo_loss)
    print "len range(nepochs)", len(range(nepochs))

    # Plot losses per epoch
    plots.plots.plot_losses_for_nf(nepochs, avg_recons_loss, avg_kl_loss, avg_elbo_loss)
    return avg_elbo_loss


def train(sess, loss_op, solver, nepochs, n_samples, batch_size,
          display_step, _X, data, summary=None, file_writer=None):
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
            with open("./output/logfile.log", "a") as f:
                f.write(line + "\n")
    print("--- %s seconds ---" % (time.time() - start_time))
    # Plot training loss per epoch
    # plt.plot(range(nepochs), avg_vae_loss)
    # plt.show()
    return avg_vae_loss


