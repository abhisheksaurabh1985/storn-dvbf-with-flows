import tensorflow as tf


class ConvolutionPlanarFlow(object):

    def __init__(self, z, z_dim=2):
        self.z = z  # shape (nSamples, nDim)
        self.z_dim = z_dim

    @staticmethod
    def softplus(x):
        return tf.log(tf.clip_by_value(x, 1e-4, 0.98))

    @staticmethod
    def tanh(x):
        return tf.tanh(x)

    @staticmethod
    def dtanh(tensor):
        return 1.0 - tf.square(tf.tanh(tensor))

    def convolution_planar_flow(self, z, flow_params, num_flows, n_latent_dim, invert_condition=True):
        pass

