
# def make_vid(X, rows=10, cols=10, h=16, w=16):
#     vid = []
#     for i in range(X.shape[1]):
#         vid.append(tile_raster_images(X[:, i, :], (h, w), (rows, cols), (1, 1)))
#     return np.array(vid)


# def create_video(SPL, save_path="/home/abhishek/Desktop/"):
#     writer = animation.FFMpegWriter(fps=15)
#
#     video = make_vid(SPL, rows=10, cols=10, h=16, w=16)
#     fig = plt.figure(figsize=(5, 5))
#     plt.axis('off')
#     frames = [[plt.imshow(frame, animated=True)] for frame in video]
#
#     vid_ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
#     if save_path is not None:
#         vid_ani.save(save_path, writer=writer)
#         plt.close()
#     return vid_ani


### Taken from breze:

# def scale_to_unit_interval(ndar, eps=1e-8):
#     """Return a copy of ndar with all values scaled between 0 and 1."""
#     ndar = ndar.copy()
#     ndar -= ndar.min()
#     ndar *= 1.0 / (ndar.max() + eps)
#     return ndar


# def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
#                        scale_rows_to_unit_interval=True,
#                        output_pixel_vals=True):
#     """Transform an array with one flattened image per row, into an array in
#     which images are reshaped and layed out like tiles on a floor.
#     This function is useful for visualizing datasets whose rows are images,
#     and also columns of matrices for transforming those rows
#     (such as the first layer of a neural net).
#     :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
#         be 2-D ndarrays or None;
#     :param X: a 2-D array in which every row is a flattened image.
#     :type img_shape: tuple; (height, width)
#     :param img_shape: the original shape of each image
#     :type tile_shape: tuple; (rows, cols)
#     :param tile_shape: the number of images to tile (rows, cols)
#     :param output_pixel_vals: if output should be pixel values (i.e. int8
#         values) or floats
#     :param scale_rows_to_unit_interval: if the values need to be scaled before
#         being plotted to [0,1] or not
#     :returns: array suitable for viewing as an image.
#         (See:`PIL.Image.fromarray`.)
#     :rtype: a 2-d array with same dtype as X.
#     """
#
#     assert len(img_shape) == 2
#     assert len(tile_shape) == 2
#     assert len(tile_spacing) == 2
#
#     # The expression below can be re-written in a more C style as
#     # follows :
#     #
#     # out_shape    = [0,0]
#     # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
#     #                tile_spacing[0]
#     # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
#     #                tile_spacing[1]
#     out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
#                  in zip(img_shape, tile_shape, tile_spacing)]
#
#     if isinstance(X, tuple):
#         assert len(X) == 4
#         # Create an output numpy ndarray to store the image
#         if output_pixel_vals:
#             out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
#         else:
#             out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)
#
#         # colors default to 0, alpha defaults to 1 (opaque)
#         if output_pixel_vals:
#             channel_defaults = [0, 0, 0, 255]
#         else:
#             channel_defaults = [0., 0., 0., 1.]
#
#         for i in range(4):
#             if X[i] is None:
#                 # if channel is None, fill it with zeros of the correct
#                 # dtype
#                 dt = out_array.dtype
#                 if output_pixel_vals:
#                     dt = 'uint8'
#                 out_array[:, :, i] = (
#                     np.zeros(out_shape, dtype=dt) + channel_defaults[i])
#             else:
#                 # use a recurrent call to compute the channel and store it
#                 # in the output
#                 out_array[:, :, i] = tile_raster_images(
#                     X[i], img_shape, tile_shape, tile_spacing,
#                     scale_rows_to_unit_interval, output_pixel_vals)
#         return out_array
#
#     else:
#         # if we are dealing with only one channel
#         H, W = img_shape
#         Hs, Ws = tile_spacing
#
#         # generate a matrix to store the output
#         dt = X.dtype
#         if output_pixel_vals:
#             dt = 'uint8'
#         out_array = np.zeros(out_shape, dtype=dt)
#
#         for tile_row in range(tile_shape[0]):
#             for tile_col in range(tile_shape[1]):
#                 if tile_row * tile_shape[1] + tile_col < X.shape[0]:
#                     if scale_rows_to_unit_interval:
#                         # if we should scale values to be between 0 and 1
#                         # do this by calling the `scale_to_unit_interval`
#                         # function
#                         this_img = scale_to_unit_interval(
#                             X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
#                     else:
#                         this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
#                     # add the slice to the corresponding position in the
#                     # output array
#                     c = 1
#                     if output_pixel_vals:
#                         c = 255
#                     out_array[
#                     tile_row * (H + Hs):tile_row * (H + Hs) + H,
#                     tile_col * (W + Ws):tile_col * (W + Ws) + W
#                     ] = this_img * c
#     return out_array

# create_video(images_arr)

# from moviepy.editor import VideoClip
#
# def make_frame(image, time_step):
#     # return images_arr[int(t), :, :, :]
#     image = np.reshape(image, (time_step, 16, 16))
#     return image[t,:,:]
#     # return images_arr[:, int(t)]
#     # return images_arr[:, int(t * (images_arr.shape[0] - 1))]
#
# animation = VideoClip(make_frame(images_arr[:,0,:], 100), duration=15) # 3-second clip
# animation.write_gif("my_animation.gif", fps=24)
#




import pickle
import numpy as np
import scipy.stats
# from tempfile import NamedTemporaryFile
# from IPython.display import HTML
# import matplotlib.pyplot as plt
# from matplotlib import animation, rc
# rc('animation', html='html5')

import os
from PIL import Image, ImageTk
from scipy.misc import imsave

# from os import listdir
from os.path import isfile, join


# from moviepy.editor import VideoClip
#
#
from data_source.dataset import Datasets, Dataset
from data_source import dataset_utils




# plt.rcParams['animation.ffmpeg_path'] = u'/usr/bin/ffmpeg'





