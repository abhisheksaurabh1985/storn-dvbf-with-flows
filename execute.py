from storn import STORN
from flows import *
import plots
from data_source.dataset import Datasets, Dataset

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

import tensorflow as tf
from tensorflow.python.framework import ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To build TF from source. Supposed to speed up the execution by 4-8x.

ops.reset_default_graph()
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
mb_size = 100
learning_rate = 0.1
training_epochs = 500
display_step = 1
# model_path = "./output_models/model.ckpt"  # Manually create the directory
# logs_path = './tf_logs/'

# Select flow type.
flow_type = "Planar"  # "Planar", "Radial", "NoFlow"
# nf_planar = True
# nf_radial = False

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

experiment_start_time = datetime.datetime.now()
output_dir = os.path.join(output_dir, experiment_start_time.strftime('%Y_%m_%d_%H_%M_%S'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Reconstruction error function. Choose either mse_reconstruction_loss or cross_entropy_loss.
reconstruction_error_function = losses.loss_functions.mse_reconstruction_loss

# Set up output model directory
if flow_type == "Planar":
    models_dir = "./output_models/planar/"
elif flow_type == "Radial":
    models_dir = "./output_models/radial/"
elif flow_type == "NoFlow":
    models_dir = "./output_models/no_flow/"

# Set up tf_logs directory
if flow_type == "Planar" or flow_type == "Radial" or flow_type == "NoFlow":
    logs_path = './tf_logs/'

# Store the parameters in a dictionary
parameters = collections.OrderedDict([('n_samples', n_samples), ('n_timesteps', n_timesteps),
                                      ('learned_reward', learned_reward), ('n_latent_dim', n_latent_dim),
                                      ('HU_enc', HU_enc), ('HU_dec', HU_dec), ('mb_size', mb_size),
                                      ('learning_rate', learning_rate), ('training_epochs', training_epochs),
                                      ('display_steps', display_step), ('flow_type', flow_type), ('numFlows', numFlows),
                                      ('apply_invertibility_conditions', apply_invertibility_condition),
                                      ('beta', beta), ('output_dir', output_dir),
                                      ('reconstruction_error_function', str(reconstruction_error_function)),
                                      ('models_dir', models_dir), ('logs_path', logs_path)])

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
nne = STORN(X_dim, n_timesteps, HU_enc, HU_dec, n_latent_dim, mb_size, learning_rate, flow_type, numFlows)
z_mu, z_logvar, flow_params = nne.encoder_rnn(_X)  # Shape:(T,B,z_dim)
z_var = tf.exp(z_logvar)

# SAMPLING
# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nne.reparametrize_z(z_mu, z_var)
print "z0 shape in execute.py:", z0.get_shape()
print "z0[0,:,:] shape in execute.py:", z0[0, :, :].get_shape()

################
# PLANAR FLOW
################
if flow_type == "Planar":
    currentClass = NormalizingPlanarFlow.NormalizingPlanarFlow(z0, n_latent_dim)


    def apply_planar_flow(previous_output, current_input):
        _z_k, _logdet_jacobian = previous_output
        print "z_k shape inside apply_planar_flow:", _z_k.get_shape()
        print "_logdet_jacobian shape inside apply_planar_flow:", _logdet_jacobian.get_shape()
        _z0, _us, _ws, _bs = current_input
        _flow_params = (_us, _ws, _bs)
        print "z0 shape inside apply_planar_flow function:", _z0.get_shape()
        _z_k, _logdet_jacobian = currentClass.planar_flow(_z0, _flow_params, numFlows, n_latent_dim,
                                                          apply_invertibility_condition)
        print "z_k shape:", _z_k.get_shape()
        print "_logdet_jacobian shape:", _logdet_jacobian.get_shape()
        return _z_k, _logdet_jacobian


    # Initialize z_k and sum_logdet_jacobian
    z_k_init = tf.zeros(shape=[tf.shape(z0)[1], tf.shape(z0)[2]], dtype=tf.float32, name="z_k_initial")
    _logdet_jacobian_init = tf.zeros(shape=[tf.shape(z0)[1], ], dtype=tf.float32, name="initial_logdet_jacobian")

    # The flow_params tuple had to be decomposed into us, ws and bs so as to make it compatible with tf.scan. Note that
    # this process is reversed inside the apply_flow function.
    print "flow_params[0]", flow_params[0].get_shape()
    print "flow_params[1]", flow_params[1].get_shape()
    print "flow_params[2]", flow_params[2].get_shape()

    us = flow_params[0]
    ws = flow_params[1]
    bs = flow_params[2]

    z_k, _logdet_jacobian = tf.scan(apply_planar_flow, (z0, us, ws, bs),
                                    initializer=(z_k_init, _logdet_jacobian_init),
                                    name="apply_flow")

    print "z_k shape in execute.py line 96ish:", z_k.get_shape()
    print "_logdet_jacobian shape in execute.py line 97ish:", _logdet_jacobian.get_shape()
    print "_logdet_jacobian", _logdet_jacobian
    sum_logdet_jacobian = tf.reduce_sum(_logdet_jacobian, axis=[0, 1])
else:
    z_k = z0

#############
# RADIAL FLOW
#############
if flow_type == "Radial":
    currentClass = NormalizingRadialFlow.NormalizingRadialFlow(z0, n_latent_dim)


    def apply_radial_flow(previous_output, current_input):
        _z_k, _logdet_jacobian = previous_output
        print "z_k shape inside apply_radial_flow:", _z_k.get_shape()
        print "_logdet_jacobian shape inside apply_radial_flow:", _logdet_jacobian.get_shape()
        _z0, _z0s, _alphas, _betas = current_input
        _flow_params = (_z0s, _alphas, _betas)
        print "z0 shape inside apply_planar_flow function:", _z0.get_shape()
        _z_k, _logdet_jacobian = currentClass.radial_flow(_z0, _flow_params, numFlows, n_latent_dim,
                                                          apply_invertibility_condition)
        print "z_k shape:", _z_k.get_shape()
        print "_logdet_jacobian shape:", _logdet_jacobian.get_shape()
        return _z_k, _logdet_jacobian


    # Initialize z_k and sum_logdet_jacobian
    z_k_init = tf.zeros(shape=[tf.shape(z0)[1], tf.shape(z0)[2]], dtype=tf.float32, name="z_k_initial")
    _logdet_jacobian_init = tf.zeros(shape=[tf.shape(z0)[1], ], dtype=tf.float32, name="initial_logdet_jacobian")

    # The flow_params tuple had to be decomposed into us, ws and bs so as to make it compatible with tf.scan. Note that
    # this process is reversed inside the apply_flow function.
    print "flow_params[0]", flow_params[0].get_shape()
    print "flow_params[1]", flow_params[1].get_shape()
    print "flow_params[2]", flow_params[2].get_shape()

    z0s = flow_params[0]
    alphas = flow_params[1]
    betas = flow_params[2]

    z_k, _logdet_jacobian = tf.scan(apply_radial_flow, (z0, z0s, alphas, betas),
                                    initializer=(z_k_init, _logdet_jacobian_init),
                                    name="apply_flow")
    sum_logdet_jacobian = tf.reduce_sum(_logdet_jacobian, axis=[0, 1])
else:
    z_k = z0

# DECODER
x_recons = nne.decoder_rnn(z_k)  # Shape: (T,B,x_dim)

# LOSS
if flow_type == "Planar":
    global_step = tf.Variable(0, trainable=False)
    loss_op, summary_losses = losses.loss_functions.elbo_loss(_X, x_recons, beta, global_step,
                                                              reconstruction_error_function, z_mu=z_mu, z_var=z_var,
                                                              z0=z0, zk=z_k, logdet_jacobian=sum_logdet_jacobian)
    # The second element of the loss_op tuple is the elbo loss.
    solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
elif flow_type == "Radial":
    global_step = tf.Variable(0, trainable=False)
    loss_op, summary_losses = losses.loss_functions.elbo_loss(_X, x_recons, beta, global_step,
                                                              reconstruction_error_function, z_mu=z_mu, z_var=z_var,
                                                              z0=z0, zk=z_k, logdet_jacobian=sum_logdet_jacobian)
    # The second element of the loss_op tuple is the elbo loss.
    solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op[2], global_step=global_step)
elif flow_type == "NoFlow":
    global_step = tf.Variable(0, trainable=False)
    # loss_op = nn_utilities.vanilla_vae_loss(_X, x_recons, z_mu, z_var)
    # loss_op, summary_losses = losses.loss_functions.mse_vanilla_vae_loss(_X, x_recons, z_mu, z_var)
    # solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    loss_op, summary_losses = losses.loss_functions.elbo_loss(_X, x_recons, z_mu=z_mu, z_var=z_var)
    solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op[2], global_step=global_step)

# Initializing the TensorFlow variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)  # Use if debugging the graph is needed.
sess.run(init)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

# Save the tensorflow model. Keep only 4 latest models. keep_checkpoint_every_n_hours can be used to save a model after
# every n hours.
saver = tf.train.Saver(max_to_keep=4)
saver.save(sess, os.path.join(models_dir, 'model.ckpt'))

# Create summary to visualise weights
# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(logs_path + "/" + str(int(time.time())),
                                    graph=sess.graph)

# TRAINING
if flow_type == "Planar":
    average_cost = train.train_nf(sess, loss_op, solver, training_epochs, n_samples, mb_size,
                                  display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
elif flow_type == "Radial":
    average_cost = train.train_nf(sess, loss_op, solver, training_epochs, n_samples, mb_size,
                                  display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
elif flow_type == "NoFlow":
    # average_cost = train.train(sess, loss_op, solver, training_epochs, n_samples, mb_size,
    #                            display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)
    average_cost = train.train_nf(sess, loss_op, solver, training_epochs, n_samples, mb_size,
                                  display_step, _X, datasets, merged_summary_op, file_writer, flow_type, output_dir)

# RECONSTRUCTION
x_sample = datasets.train.next_batch(mb_size)
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

plots.plots.plot_signals_and_reconstructions(time_steps, actual_signals, recons_signals, flow_type, output_dir,
                                             points_to_plot)

sess.close()
