import numpy as np

def sliceFrom3DTensor(tensor, idx):
    """
    Returns a 2D slice from a 3D tensor slicing along the third dimension.
    Assumes the tensor to be of shape (T*B*D).
    
    Args:
        tensor: 3D tensor to be sliced.
        idx: Index of the 
        
    Returns:
        tensor_2D:
    """
    tensor_2D = tensor[:, :, idx]
    return tensor_2D


def remove_outliers(data, m=2.):
    """
    Modified z-score. Refer this link: http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    :param data:
    :param m:
    :return:
    """
    d = np.abs(data - np.median(data, axis=0))
    print "d shape", d.shape
    mdev = np.median(d, axis=0)
    print "mdev shape", mdev.shape
    s = d / mdev if np.all(mdev) else 0.
    print s.shape
    data_sans_outlier = data[np.where(s[:, 3] < m)]
    print "data_sans outlier shape", data_sans_outlier.shape
    return data_sans_outlier


def read_loss_logfile(fname):
    """
    Read logfile.log to get the losses. Returns a numpy array with number of epochs and all three losses.
    """
    phrases = ['Epoch:', 'Average recons loss:', 'Average kl loss:', 'Average elbo loss:']
    nepoch , avg_recons_loss , avg_kl_loss , avg_elbo_loss = [], [], [], []
    with open(fname) as f:
        for line in f:
            for x in phrases:
                line = line.replace(x, " ")
            words = line.split()

            if len(words) > 0:
                if words[0] == "Experiment":
                    continue
                nepoch.append(words[0])
                avg_recons_loss.append(float(words[1]))
                avg_kl_loss.append(float(words[2]))
                avg_elbo_loss.append(float(words[3]))
    num_epochs = np.expand_dims(np.asarray(nepoch), 1)
    recons_losses = np.expand_dims(np.asarray(avg_recons_loss), 1)
    kl_losses = np.expand_dims(np.asarray(avg_kl_loss), 1)
    elbo_losses = np.expand_dims(np.asarray(avg_elbo_loss), 1)
    losses_per_epoch = np.concatenate((num_epochs, recons_losses, kl_losses, elbo_losses), axis=1).astype(np.float32)
    return losses_per_epoch



