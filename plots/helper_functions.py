import numpy as np
import scipy.stats
# from tempfile import NamedTemporaryFile
# from IPython.display import HTML
import matplotlib.pyplot as plt
# from matplotlib import animation, rc
# rc('animation', html='html5')
import os
from scipy.misc import imsave


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

    As m increases, less and less outliers are removed.

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
    nepoch, avg_recons_loss, avg_kl_loss, avg_elbo_loss = [], [], [], []
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


def get_obs(_obs):
    """
    cos and sine theta as inputs.
    :param _obs:
    :return:
    """
    image_dims = [16, 16]
    x = np.linspace(-4, 4, image_dims[0])
    y = np.linspace(-4, 4, image_dims[1])
    xv, yv = np.meshgrid(x, y)
    r = np.array([[_obs[0], -_obs[1]], [_obs[1], _obs[0]]])

    obs = scipy.stats.norm.pdf(np.dot(np.concatenate((xv.ravel()[:, np.newaxis], yv.ravel()[:, np.newaxis]), 1), r),
                                loc=[0, 2.0], scale=[0.5, 0.9]).prod(1)
    obs += np.random.randn(*obs.shape) * 0.01
    return obs


def make_images(data, output_dir, img_width=16, img_height=16):
    """
    Create images from numpy array and save it for generating GIFs.
    """
    print data.shape[0]
    time_steps = range(data.shape[0])
    for time_step in time_steps:
        print time_step
        img_arr = np.reshape(data[time_step, :], (img_width, img_height))
        # img = Image.fromarray(img_arr, 'RGB')
        # output_dir = os.path.join(output_dir, 'ts_' + str(time_step) + '.png')
        # print output_dir
        # print type(output_dir)
        # img.save(output_dir)
        imsave(os.path.join(output_dir, 'ts_' + str(time_step) + '.png'), img_arr)
        # img.show()


def generate_gif(dirname):
    """
    Generate gif from images
    :param dirname:
    :return:
    """
    os.system('convert -delay 50 -loop 0 {0}/ts_*png {0}/gs.gif'.format(dirname))
    return






