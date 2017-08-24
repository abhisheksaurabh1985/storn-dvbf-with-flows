import numpy as np
import pickle
import os
from os.path import isfile, join
import numpy as np
import scipy.stats
from PIL import Image
from scipy.misc import imsave


from data_source.dataset import Datasets, Dataset
from data_source import dataset_utils


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
    time_steps = range(data.shape[0])
    for time_step in time_steps:
        img_arr = np.reshape(data[time_step, :], (img_width, img_height))
        # img = Image.fromarray(img_arr, 'RGB')
        # output_dir = os.path.join(output_dir, 'ts_' + str(time_step) + '.png')
        # print output_dir
        # print type(output_dir)
        # img.save(output_dir)
        imsave(os.path.join(output_dir, 'ts_' + str(time_step) + '.png'), img_arr)
        # img.show()


def generate_gif(input_dir, output_dir):
    """
    Generate gif from images
    :param dirname:
    :return:
    """
    # os.system('convert -delay 50 -loop 0 {0}/ts_*png {0}/gs.gif'.format(input_dir))
    os.system('convert -delay 50 -loop 0 {0}ts_*png {1}actual_data_as_image_grid.gif'.format(input_dir, output_dir))
    return


"""
Plot of images of the actual signal
"""

# Set up directories
dir_actual_data_as_image = "./output/actual_data_as_image/"
if not os.path.exists(dir_actual_data_as_image):
    os.makedirs(dir_actual_data_as_image)

dir_gif_from_actual_data = "./output/actual_data_as_image/gif/"
if not os.path.exists(dir_gif_from_actual_data):
    os.makedirs(dir_gif_from_actual_data)

dir_image_grid_from_actual_data = "./output/actual_data_as_image/gif/"
if not os.path.exists(dir_image_grid_from_actual_data):
    os.makedirs(dir_image_grid_from_actual_data)

file_actual_data_as_image_grid = os.path.join(dir_image_grid_from_actual_data, "/" + "actual_data_image_grid.png")

datasets = pickle.load(open('./pickled_data/datasets.pkl', "rb"))
data = datasets.train.next_batch(99)

# data = pickle.load(open('./pickled_data/junk_gs_samples.pkl', "rb"))

batch_size = data.shape[1]
# Get cosine and sine from the dataset to create an array
images = []
for co, si in data[:, :, :2].reshape((-1, 2)):
    image = get_obs([co, si])  # image.shape:(256,)
    images.append(image)  # Final len(images)= 10000

images_arr = np.array(images).reshape((100, batch_size, -1))  # images_arr shape: (100,100,256)

make_images(images_arr[:, 0, :], dir_actual_data_as_image)  # For the 0th instance in the batch.
generate_gif(dir_actual_data_as_image, dir_gif_from_actual_data)


# https://stackoverflow.com/questions/20038648/writting-a-file-with-multiple-images-in-a-grid
files = [f for f in os.listdir(dir_actual_data_as_image) if isfile(join(dir_actual_data_as_image, f))]

fpath_actual_data_as_image_grid = os.path.join(dir_image_grid_from_actual_data, "actual_data_image_grid.png")
new_im = Image.new('L', (960, 960))

# from PIL import ImageDraw
# draw = ImageDraw.Draw(new_im)
# draw.line((0, 16, 32, 48, 64, 80, 96, 112, 128, 134, 144, 160), fill=128, width=10)


index = 0
for i in xrange(0, 960, 96):
    for j in xrange(0, 960, 96):
        im = Image.open(os.path.join(dir_actual_data_as_image, files[index]))
        im.thumbnail((16, 16))
        new_im.paste(im, (i, j))
        index += 1

new_im.save(fpath_actual_data_as_image_grid)
