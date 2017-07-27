import tensorflow as tf


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
    with tf.summary.scalar('min_max'):
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def inputs(D, Z, time_steps):
    """
    D: Input dimension
    Z: Latent space dimension
    """
    X = tf.placeholder(tf.float32, shape=[time_steps, None, D],
                       name='input_data')
    z = tf.placeholder(tf.float32, shape=[time_steps, None, Z],
                       name='latent_var')
    return X, z


def initialize_weights_random_normal(dim1, dim2, mu, sigma):
    return tf.random_normal((dim1, dim2), mean=mu, stddev=sigma)


def initialize_bias_with_zeros(dim):
    return tf.zeros((dim, 1))


