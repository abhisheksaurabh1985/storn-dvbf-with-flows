from matplotlib import pyplot as plt


def line_plot_2d(x, y):
    plt.plot(x, y)
    return


def plot_signals_and_reconstructions(time_steps, actual, recons):
    # Four axes, returned as a 2-d array
    f, axarr = plt.subplots(2, 2)
    f.suptitle("Actual and Reconstructed Signals", fontsize="x-large")

    axarr[0, 0].plot(time_steps, actual[0][:, 0])
    axarr[0, 0].plot(time_steps, recons[0][:, 0])
    axarr[0, 0].set_title('Cosine')

    axarr[0, 1].plot(time_steps, actual[1][:, 0])
    axarr[0, 1].plot(time_steps, recons[1][:, 0])
    axarr[0, 1].set_title('Sine')

    axarr[1, 0].plot(time_steps, actual[2][:, 0])
    axarr[1, 0].plot(time_steps, recons[2][:, 0])
    axarr[1, 0].set_title('Velocity')

    axarr[1, 1].plot(time_steps, actual[3][:, 0])
    axarr[1, 1].plot(time_steps, recons[3][:, 0])
    axarr[1, 1].set_title('Reward')

    legend_labels = ["Actual", "Reconstruction"]
    plt.legend(legend_labels, ncol=len(legend_labels), title="Legend")
    f.tight_layout()
    f.savefig("./output/actual_reconstructed_signals.png")
    plt.show()
