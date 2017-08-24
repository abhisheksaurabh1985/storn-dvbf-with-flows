import numpy as np
import scipy

from tempfile import NamedTemporaryFile
from IPython.display import HTML

import matplotlib.pyplot as plt
from matplotlib import animation, rc

rc('animation', html='html5')


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
                                loc=[0, -2.0], scale=[0.5, 0.9]).prod(1)
    obs += np.random.randn(*obs.shape) * 0.01
    return obs


def make_vid(X, rows=10, cols=10, h=16, w=16):
    vid = []
    for i in range(X.shape[1]):
        vid.append(tile_raster_images(X[:, i, :], (h, w), (rows, cols), (1, 1)))
    return np.array(vid)


def create_video(SPL, save_path=None):
    writer = animation.FFMpegWriter(fps=15)

    video = make_vid(SPL, rows=10, cols=10, h=16, w=16)
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    frames = [[plt.imshow(frame, animated=True)] for frame in video]

    vid_ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    if save_path is not None:
        vid_ani.save(save_path, writer=writer)
        plt.close()
    return vid_ani


# Taken from breze:

def scale_to_unit_interval(ndar, eps=1e-8):
    """Return a copy of ndar with all values scaled between 0 and 1."""
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
        be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.
    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image
    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)
    :param output_pixel_vals: if output should be pixel values (i.e. int8
        values) or floats
    :param scale_rows_to_unit_interval: if the values need to be scaled before
        being plotted to [0,1] or not
    :returns: array suitable for viewing as an image.
        (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = (
                    np.zeros(out_shape, dtype=dt) + channel_defaults[i])
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs):tile_row * (H + Hs) + H,
                    tile_col * (W + Ws):tile_col * (W + Ws) + W
                    ] = this_img * c
    return out_array

import os
import pickle

# Set up directories
dir_gs_data_as_image = "/home/abhishek/Desktop/Junk/test_gs/"
if not os.path.exists(dir_gs_data_as_image):
    os.makedirs(dir_gs_data_as_image)

dir_gif_from_data = "/home/abhishek/Desktop/Junk/test_gs/gif/"
if not os.path.exists(dir_gif_from_data):
    os.makedirs(dir_gif_from_data)

dir_image_grid_from_data = "./output/actual_data_as_image/gif/"
if not os.path.exists(dir_image_grid_from_data):
    os.makedirs(dir_image_grid_from_data)

file_actual_data_as_image_grid = os.path.join(dir_image_grid_from_data, "/" + "gs_as_image_grid.png")

# data = pickle.load(open('/home/abhishek/Desktop/Junk/test_gs/gs_samples.pkl', "rb"))
data = pickle.load(open('/home/abhishek/Desktop/Junk/test_gs/datasets.pkl', "rb"))
data = data.train.next_batch(100)

# data = pickle.load(open('./pickled_data/junk_gs_samples.pkl', "rb"))

batch_size = data.shape[1]
# Get cosine and sine from the dataset to create an array
images = []
for co, si in data[:, :, :2].reshape((-1, 2)):
    image = get_obs([co, si])  # image.shape:(256,)
    images.append(image)  # Final len(images)= 10000

images_arr = np.array(images).reshape((100, batch_size, -1))  # images_arr shape: (100,100,256)


"""
Following is done to get a batch size of 100. Code for creating video and image grid only works for a batch size of 100.
"""
images_arr = np.concatenate([images_arr, images_arr, images_arr, images_arr, images_arr], axis=1)

create_video(images_arr, dir_gif_from_data)