from storn import STORN
from flows import *
import plots
from data_source.dataset import Datasets, Dataset
from data_source import dataset_utils

# from dataset import *  # Needed despite using the pickled files!
import nn_utilities
import train
# import helper_functions
# import plots
import losses

import os
import pickle
import time
import datetime
import json
import collections
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To build TF from source. Supposed to speed up the execution by 4-8x.

ops.reset_default_graph()
tf.set_random_seed(1234)
# from tensorflow.python import debug as tf_debug
# from matplotlib import pyplot as plt


# Dataset parameters
n_samples = 1000
n_timesteps = 100
learned_reward = True  # is the reward handled as observation?
# NN params
n_latent_dim = 2
HU_enc = 128
HU_dec = 128
mb_size = 20
learning_rate = 0.0001  # 0.0001 for Planar works well.
training_epochs = 3
display_step = 1
mu_init = 0  # Params for random normal weight initialization
sigma_init = 0.001  # Params for random normal weight initialization
decoder_output_function = tf.identity
activation_function = tf.nn.relu
# model_path = "./output_models/model.ckpt"  # Manually create the directory
# logs_path = './tf_logs/'

# Select flow type.
flow_type = "NoFlow"  # "ConvolutionPlanar", "Planar", "Radial", "NoFlow"

# Flow parameters
numFlows = 4  # Number of times flow has to be applied.
apply_invertibility_condition = True
beta = True

# Plot parameters
points_to_plot = [0, 2, 4, 6, 8, 10]  # Points in the mini batches which are to be reconstructed and plotted

# Set up output directory
if flow_type == "Planar":
    output_dir = "./output/planar/"
elif flow_type == "Radial":
    output_dir = "./output/radial/"
elif flow_type == "NoFlow":
    output_dir = "./output/no_flow/"
elif flow_type == "ConvolutionPlanar":
    output_dir = "./output/conv_planar/"

experiment_start_time = datetime.datetime.now()
output_dir = os.path.join(output_dir, experiment_start_time.strftime('%Y_%m_%d_%H_%M_%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Reconstruction error function. Choose either negative_log_normal, mse_reconstruction_loss or cross_entropy_loss.
reconstruction_error_function = losses.loss_functions.negative_log_normal

# Restore saved model
restore_model = False
# If restore_model is True, set path to the model to be restored. Ignore file name model.ckpt.
fpath_restore_model = "./output/planar/2017_08_08_12_02_37/output_models/"

# Set up output model directory
if flow_type == "Planar":
    models_dir = os.path.join(output_dir, "output_models/")
elif flow_type == "Radial":
    models_dir = os.path.join(output_dir, "output_models/")
elif flow_type == "NoFlow":
    models_dir = os.path.join(output_dir, "output_models/")
elif flow_type == "ConvolutionPlanar":
    models_dir = os.path.join(output_dir, "output_models/")

# Create directory for output models
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Set up tf_logs directory
if flow_type == "Planar" or flow_type == "Radial" or flow_type == "NoFlow" or flow_type == "ConvolutionPlanar":
    logs_path = './tf_logs/'

# Optimizer
optimizer = tf.train.AdamOptimizer

# Normalize data
normalize_data = False

# Output directory for probability distribution and generative samples
dir_pdist = os.path.join(output_dir, "pdist/")
if not os.path.exists(dir_pdist):
    os.makedirs(dir_pdist)
dir_gs = os.path.join(output_dir, "gs/")
if not os.path.exists(dir_gs):
    os.makedirs(dir_gs)

filter_width = 3

dir_actual_data_as_image = "./actual_data_as_image/"
if not os.path.exists(dir_actual_data_as_image):
    os.makedirs(dir_actual_data_as_image)

dir_gif_from_actual_data = "./actual_data_as_image/gif/"
if not os.path.exists(dir_gif_from_actual_data):
    os.makedirs(dir_gif_from_actual_data)


dir_image_grid_from_actual_data = "./actual_data_as_image/gif/"
if not os.path.exists(dir_image_grid_from_actual_data):
    os.makedirs(dir_image_grid_from_actual_data)

dir_saved_variables = os.path.join(output_dir, "saved_vars/")
if not os.path.exists(dir_saved_variables):
    os.makedirs(dir_saved_variables)

model_type = "storn_without_input"

# Store the parameters in a dictionary
parameters = collections.OrderedDict([('n_samples', n_samples), ('n_timesteps', n_timesteps),
                                      ('learned_reward', learned_reward), ('n_latent_dim', n_latent_dim),
                                      ('HU_enc', HU_enc), ('HU_dec', HU_dec), ('mb_size', mb_size),
                                      ('learning_rate', learning_rate), ('training_epochs', training_epochs),
                                      ('display_steps', display_step),
                                      ('mu_init', mu_init), ('sigma_init', sigma_init),
                                      ('flow_type', flow_type), ('numFlows', numFlows),
                                      ('decoder_output_function', str(decoder_output_function)),
                                      ('activation_function', str(activation_function)),
                                      ('apply_invertibility_conditions', apply_invertibility_condition),
                                      ('beta', beta), ('output_dir', output_dir),
                                      ('restore_model', restore_model),
                                      ('fpath_restore_model', fpath_restore_model),
                                      ('reconstruction_error_function', str(reconstruction_error_function)),
                                      ('models_dir', models_dir), ('logs_path', logs_path),
                                      ('optimizer', str(optimizer)),
                                      ('normalize_data', normalize_data),
                                      ('dir_pdist', dir_pdist),
                                      ('dir_gs', dir_gs), ('filter_width', filter_width),
                                      ('dir_actual_data_as_image', dir_actual_data_as_image),
                                      ('dir_gif_from_actual_data', dir_gif_from_actual_data),
                                      ('dir_image_grid_from_actual_data', dir_image_grid_from_actual_data),
                                      ('model_type', model_type)])

# Write hyper-parameters with time-stamp in a file. Also write the same time stamp in the logfile.log
# experiment_start_time = time.strftime("%c")
with open(os.path.join(output_dir, "parameters.log"), "a") as f:
    f.write("Experiment start time:" + experiment_start_time.strftime('%d %b %Y %H:%M:%S') + "\n")
    f.write(json.dumps(parameters, indent=4))
    f.write("\n")

with open(os.path.join(output_dir, "logfile.log"), "a") as f:
    f.write("\n" + "Experiment start time:" + experiment_start_time.strftime('%d %b %Y %H:%M:%S') + "\n\n")

# DATASET
XU = pickle.load(open('./pickled_data/XU.pkl', "rb"))
shuffled_data = pickle.load(open('./pickled_data/shuffled_data.pkl', "rb"))
datasets = pickle.load(open('./pickled_data/datasets.pkl', "rb"))

# ENCODER
X_dim = datasets.train.full_data.shape[2]  # Input data dimension
_X, z = nn_utilities.inputs(X_dim, n_latent_dim, n_timesteps)
nne = STORN(X_dim, n_timesteps, HU_enc, HU_dec, n_latent_dim, mb_size, learning_rate, flow_type, numFlows,
            mu_init, sigma_init, decoder_output_function, activation_function)
z_mu, z_logvar, flow_params = nne.encoder_rnn(_X)  # Shape:(T,B,z_dim)
z_var = tf.exp(z_logvar)
print "###########################@@@@@@@@@@@@@@@@@@@@@ z_mu shape", z_mu.get_shape()
print "###########################@@@@@@@@@@@@@@@@@@@@@ z_var shape", z_var.get_shape()

# SAMPLING
# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nne.reparametrize_z(z_mu, z_var)

# Apply flow
if flow_type == "Planar":
    currentClass = NormalizingPlanarFlow.NormalizingPlanarFlow(z0, n_latent_dim)


    def apply_planar_flow(previous_output, current_input):
        _z_k, _logdet_jacobian = previous_output
        _z0, _us, _ws, _bs = current_input
        _flow_params = (_us, _ws, _bs)
        _z_k, _logdet_jacobian = currentClass.planar_flow(_z0, _flow_params, numFlows, n_latent_dim,
                                                          apply_invertibility_condition)
        return _z_k, _logdet_jacobian


    # Initialize z_k and sum_logdet_jacobian
    z_k_init = tf.zeros(shape=[tf.shape(z0)[1], tf.shape(z0)[2]], dtype=tf.float32, name="z_k_initial")
    _logdet_jacobian_init = tf.zeros(shape=[tf.shape(z0)[1], ], dtype=tf.float32, name="initial_logdet_jacobian")

    # The flow_params tuple had to be decomposed into us, ws and bs so as to make it compatible with tf.scan. Note that
    # this process is reversed inside the apply_flow function.
    us = flow_params[0]
    ws = flow_params[1]
    bs = flow_params[2]

    z_k, _logdet_jacobian = tf.scan(apply_planar_flow, (z0, us, ws, bs),
                                    initializer=(z_k_init, _logdet_jacobian_init),
                                    name="apply_flow")
    # sum_logdet_jacobian = tf.reduce_sum(_logdet_jacobian, axis=[0, 1])
    # sum_logdet_jacobian = tf.reduce_sum(_logdet_jacobian, axis=1)
    print "############### _logdet_jacobian", _logdet_jacobian.get_shape()
    sum_logdet_jacobian = _logdet_jacobian
elif flow_type == "Radial":
    currentClass = NormalizingRadialFlow.NormalizingRadialFlow(z0, n_latent_dim)


    def apply_radial_flow(previous_output, current_input):
        _z_k, _logdet_jacobian = previous_output
        _z0, _z0s, _alphas, _betas = current_input
        _flow_params = (_z0s, _alphas, _betas)
        _z_k, _logdet_jacobian = currentClass.radial_flow_modified(_z0, _flow_params, numFlows, n_latent_dim,
                                                                   apply_invertibility_condition)
        return _z_k, _logdet_jacobian


    # Initialize z_k and sum_logdet_jacobian
    z_k_init = tf.zeros(shape=[tf.shape(z0)[1], tf.shape(z0)[2]], dtype=tf.float32, name="z_k_initial")
    _logdet_jacobian_init = tf.zeros(shape=[tf.shape(z0)[1], ], dtype=tf.float32, name="initial_logdet_jacobian")

    # The flow_params tuple had to be decomposed into us, ws and bs so as to make it compatible with tf.scan. Note that
    # a tuple is again created inside the apply_flow function.
    z0s = flow_params[0]
    alphas = flow_params[1]
    betas = flow_params[2]

    z_k, _logdet_jacobian = tf.scan(apply_radial_flow, (z0, z0s, alphas, betas),
                                    initializer=(z_k_init, _logdet_jacobian_init),
                                    name="apply_flow")
    # sum_logdet_jacobian = tf.reduce_sum(_logdet_jacobian, axis=[0, 1])
    sum_logdet_jacobian = _logdet_jacobian
elif flow_type == "ConvolutionPlanar":
    currentClass = ConvolutionPlanarFlow.ConvolutionPlanarFlow(z0, n_latent_dim)
    print "z0 shape:", z0.get_shape()
    print "z0 transposed shape:", tf.transpose(z0, perm=[1, 0, 2]).get_shape()
    z_k, sum_logdet_jacobian = currentClass.convolution_planar_flow(tf.transpose(z0, perm=[1, 0, 2]),
                                                                    flow_params, numFlows, n_latent_dim,
                                                                    filter_width=3)
elif flow_type == "NoFlow":
    z_k = z0

# DECODER
mu_x_recons, logvar_x_recons = nne.decoder_rnn(z_k, model_type, input_x=_X)  # Shape: (T,B,x_dim)
var_x_recons = tf.exp(logvar_x_recons)
x_recons = nne.reparametrize_z(mu_x_recons, var_x_recons)

# LOSS
if flow_type == "Planar":
    global_step = tf.Variable(0, trainable=False)
    loss_op, summary_losses, probability_distributions = \
        losses.loss_functions.elbo_loss(_X, x_recons, beta, global_step,
                                        reconstruction_error_function, z_mu=z_mu, z_var=z_var,
                                        z0=z0, zk=z_k, logdet_jacobian=sum_logdet_jacobian,
                                        decoder_mean=mu_x_recons, decoder_variance=var_x_recons)
    # The second element of the loss_op tuple is the elbo loss.
    solver = optimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
    # solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
    # solver = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
elif flow_type == "ConvolutionPlanar":
    global_step = tf.Variable(0, trainable=False)
    loss_op, summary_losses, probability_distributions = \
        losses.loss_functions.elbo_loss(_X, x_recons, beta, global_step,
                                        reconstruction_error_function, z_mu=z_mu, z_var=z_var,
                                        z0=z0, zk=z_k, logdet_jacobian=sum_logdet_jacobian,
                                        decoder_mean=mu_x_recons, decoder_variance=var_x_recons)
    # The second element of the loss_op tuple is the elbo loss.
    solver = optimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
elif flow_type == "Radial":
    global_step = tf.Variable(0, trainable=False)
    loss_op, summary_losses, probability_distributions = \
        losses.loss_functions.elbo_loss(_X, x_recons, beta, global_step,
                                        reconstruction_error_function, z_mu=z_mu, z_var=z_var,
                                        z0=z0, zk=z_k, logdet_jacobian=sum_logdet_jacobian,
                                        decoder_mean=mu_x_recons, decoder_variance=var_x_recons)
    # The second element of the loss_op tuple is the elbo loss.
    solver = optimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
elif flow_type == "NoFlow":
    global_step = tf.Variable(0, trainable=False)
    # loss_op = nn_utilities.vanilla_vae_loss(_X, x_recons, z_mu, z_var)
    # loss_op, summary_losses = losses.loss_functions.mse_vanilla_vae_loss(_X, x_recons, z_mu, z_var)
    # solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    loss_op, summary_losses, probability_distributions = \
        losses.loss_functions.elbo_loss(_X, x_recons, beta, global_step,
                                        reconstruction_error_function, z_mu=z_mu, z_var=z_var,
                                        decoder_mean=mu_x_recons, decoder_variance=var_x_recons)
    solver = optimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
    # solver = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_op[2], global_step=global_step)

# Initializing the TensorFlow variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)  # Use if debugging the graph is needed.
sess.run(init)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

# Create summary to visualise weights
# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logs_path + "/" + str(int(time.time())),
                                    graph=sess.graph)

saver = tf.train.Saver(max_to_keep=4)

if not restore_model:

    # TRAINING
    if flow_type == "Planar":
        average_cost = train.train_nf(sess, loss_op, summary_losses, probability_distributions,
                                      solver, training_epochs, n_samples, mb_size,
                                      display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
    elif flow_type == "ConvolutionPlanar":
        average_cost = train.train_nf(sess, loss_op, summary_losses, probability_distributions,
                                      solver, training_epochs, n_samples, mb_size,
                                      display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
    elif flow_type == "Radial":
        average_cost = train.train_nf(sess, loss_op, summary_losses, probability_distributions,
                                      solver, training_epochs, n_samples, mb_size,
                                      display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
    elif flow_type == "NoFlow":
        # average_cost = train.train(sess, loss_op, solver, training_epochs, n_samples, mb_size,
        #                            display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
        average_cost = train.train_nf(sess, loss_op, summary_losses, probability_distributions,
                                      solver, training_epochs, n_samples, mb_size,
                                      display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)

    # Save the tensorflow model. Keep only 4 latest models. keep_checkpoint_every_n_hours can be used to save a model
    # after every n hours.
    saver.save(sess, os.path.join(models_dir, 'model.ckpt'))
else:
    saver.restore(sess, os.path.join(fpath_restore_model, 'model.ckpt'))


# RECONSTRUCTION
x_sample = datasets.train.next_batch(mb_size)
if normalize_data:
    x_sample = dataset_utils.normalize_data(x_sample)

print "x_sample.shape", x_sample.shape

# latent_for_x_sample = nne.get_latent(sess, _X, x_sample)
# print "latent sample shape", latent_for_x_sample.shape

x_reconstructed = nne.reconstruct(sess, _X, x_sample)
print "x_reconstructed type", type(x_reconstructed)
print "x_reconstructed shape", x_reconstructed.shape

# PLOTS
# Prepare data for plotting
cos_actual = plots.helper_functions.sliceFrom3DTensor(x_sample, 0)
sine_actual = plots.helper_functions.sliceFrom3DTensor(x_sample, 1)
w_actual = plots.helper_functions.sliceFrom3DTensor(x_sample, 2)  # Angular velocity omega
reward_actual = plots.helper_functions.sliceFrom3DTensor(x_sample, 3)
print cos_actual.shape, sine_actual.shape, w_actual.shape, reward_actual.shape

cos_recons = plots.helper_functions.sliceFrom3DTensor(x_reconstructed, 0)
sine_recons = plots.helper_functions.sliceFrom3DTensor(x_reconstructed, 1)
w_recons = plots.helper_functions.sliceFrom3DTensor(x_reconstructed, 2)  # Angular velocity omega
reward_recons = plots.helper_functions.sliceFrom3DTensor(x_reconstructed, 3)
print cos_recons.shape, sine_recons.shape, w_recons.shape, reward_recons.shape

# Plot cosine: actual, reconstruction and generative sampling
time_steps = range(n_timesteps)
actual_signals = [cos_actual, sine_actual, w_actual, reward_actual]
recons_signals = [cos_recons, sine_recons, w_recons, reward_recons]
plots.plots.plot_signals_and_reconstructions(time_steps, actual_signals, recons_signals, flow_type,
                                             output_dir, points_to_plot)

# Dump time_steps, actual and reconstructed signals
pickle.dump([x_sample, x_reconstructed, time_steps, actual_signals, recons_signals],
            open(os.path.join(dir_saved_variables, 'saved_vars.pickle'), "wb"))

# Plot probability distributions of the reconstruction and actual
plots.plots.distribution_signals(x_sample, dir_pdist, flow_type, signal="actual")
plots.plots.distribution_signals(x_reconstructed, dir_pdist, flow_type, signal="recons")

# GENERATIVE SAMPLES
def latent_standard_normal_prior(nts, mbs, zdim):
    mean = tf.zeros(shape=[nts, mbs, zdim], dtype=tf.float32,
                    name="z_mu_generative_sampling")
    variance = tf.ones(shape=[nts, mbs, zdim], dtype=tf.float32,
                       name="z_var_generative_sampling")
    latent_var = nne.reparametrize_z(mean, variance)
    return mean, variance, latent_var

if model_type == "storn_with_input":
    gs_z_mu, gs_z_var, gs_z0 = latent_standard_normal_prior(n_timesteps, mb_size, n_latent_dim)

    gs_x_init = tf.random_normal([100, X_dim], name="gs_x_init", dtype=tf.float32)  # Initialize x

    gs_mu_x_recons, gs_logvar_x_recons = nne.decoder_rnn(gs_z0, model_type, input_x=gs_x_init,
                                                         operation_type='generative_sampling')
    gs_var_x_recons = tf.exp(gs_logvar_x_recons)
    gs_x_recons = nne.reparametrize_z(gs_mu_x_recons, gs_var_x_recons)


    def generative_samples(sess, gs_x_recons):
        return sess.run(gs_x_recons)

    gs_samples = generative_samples(sess, gs_x_recons)
    print "gs_samples shape:", gs_samples.shape
    pickle.dump(gs_samples, open(os.path.join(dir_gs, 'gs_samples.pkl'), "wb"))
elif model_type == "storn_without_input":
    gs_z_mu, gs_z_var, gs_z0 = latent_standard_normal_prior(n_timesteps, mb_size, n_latent_dim)

    gs_mu_x_recons, gs_logvar_x_recons = nne.decoder_rnn(gs_z0, model_type)
    gs_var_x_recons = tf.exp(gs_logvar_x_recons)
    gs_x_recons = nne.reparametrize_z(gs_mu_x_recons, gs_var_x_recons)


    def generative_samples(sess, gs_x_recons):
        return sess.run(gs_x_recons)

    gs_samples = generative_samples(sess, gs_x_recons)
    print "gs_samples shape:", gs_samples.shape
    pickle.dump(gs_samples, open(os.path.join(dir_gs, 'gs_samples.pkl'), "wb"))

sess.close()


# cos_generative_sample = plots.helper_functions.sliceFrom3DTensor(gs_samples, 0)
# gs_images_array = plots.helper_functions.get_obs(cos_generative_sample)
# print "gs_images_array shape:", gs_images_array.shape
# gs_images_array_prime = np.reshape(gs_images_array, (mb_size, 100, X_dim))
#
# plots.helper_functions.create_video(gs_images_array_prime, save_path="/home/abhishek/Desktop/")

# cos_generative_sample = plots.helper_functions.sliceFrom3DTensor(gs_samples, 0)
# sine_generative_sample = plots.helper_functions.sliceFrom3DTensor(gs_samples, 1)
# w_generative_sample = plots.helper_functions.sliceFrom3DTensor(gs_samples, 2)  # Angular velocity omega
# reward_generative_sample = plots.helper_functions.sliceFrom3DTensor(gs_samples, 3)
# print cos_generative_sample.shape, sine_generative_sample.shape, w_generative_sample.shape, reward_generative_sample.shape

# generative_signals = [cos_generative_sample, sine_generative_sample, w_generative_sample, reward_generative_sample]


# plots.plots.plot_generative_samples(time_steps, generative_signals, flow_type, output_dir)

# gs_cos = np.concatenate((np.expand_dims(cos_generative_sample[:, 0], 1),
#                          np.expand_dims(range(n_timesteps), 1)), axis=1)
# print gs_cos.shape

# plots.plots.distribution_signals(gs_samples, dir_pdist, flow_type, signal="gs")


