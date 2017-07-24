import time
# from matplotlib import pyplot as plt


def train(sess, loss_op, solver, nepochs, n_samples, batch_size,
          display_step, _X, data, summary=None, file_writer=None):
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
