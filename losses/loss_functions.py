import tensorflow as tf


def mse_reconstruction_loss(x, x_reconstr):
    with tf.name_scope("mse_reconstruction_loss"):
        reconstr_loss = tf.reduce_sum((x - x_reconstr) ** 2, name="mse_reconstruction_loss")
        return reconstr_loss


def cross_entropy_loss(prediction, actual, offset=1e-4):
    with tf.name_scope("cross_entropy_loss"):
        _prediction = tf.clip_by_value(prediction, offset, 1 - offset)
        ce_loss = - tf.reduce_sum(actual * tf.log(_prediction)
                                  + (1 - actual) * tf.log(1 - _prediction), 1, name="cross_entropy_loss")
        return ce_loss


def kl_divergence_gaussian(mu, var):
    with tf.name_scope("kl_divergence"):
        # kl = - 0.5 * tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), 1)
        _var = tf.clip_by_value(var, 1e-4, 1e6)
        kl = - 0.5 * tf.reduce_sum(1 + tf.log(_var) - tf.square(mu) - tf.exp(tf.log(_var)), 1,
                                   name="kl_divergence_gaussian")
        return kl


def gaussian_log_pdf(z, mu, var):
    with tf.name_scope("gaussian_log_pdf"):
        return tf.contrib.distributions.MultivariateNormalDiag(
            loc=mu, scale_diag=tf.maximum(tf.sqrt(var), 1e-4)).log_prob(z + 1e-4)


def elbo_loss(actual, prediction, beta=True, global_step=tf.Variable(0, trainable=False),
              recons_error_func=mse_reconstruction_loss, **kwargs):
    mu = kwargs['z_mu']
    _var = kwargs['z_var']

    if 'logdet_jacobian' not in kwargs:
        recons_loss = tf.reduce_sum((actual - prediction) ** 2, name="reconstruction_loss")
        kl_loss = -0.5 * tf.reduce_sum(1 + tf.log(1e-6 + _var) - tf.square(mu) - _var,
                                       axis=None, name="kl_loss")
        _elbo_loss = tf.reduce_mean(recons_loss + kl_loss, name="elbo_loss")

        # Summary of losses
        tf.summary.scalar("reconstruction_loss", recons_loss)
        tf.summary.scalar("latent_loss", kl_loss)
        tf.summary.scalar("elbo_loss", _elbo_loss)
        merged_summary_losses = tf.summary.merge_all()
        return (recons_loss, kl_loss, _elbo_loss), merged_summary_losses
    else:
        z0 = kwargs['z0']
        zk = kwargs['zk']
        logdet_jacobian = kwargs['logdet_jacobian']

        # First term
        log_q0_z0 = tf.reduce_sum(gaussian_log_pdf(z0, mu, _var))
        print "log_q0_z0 shape: ", log_q0_z0.get_shape()
        # Third term
        # sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian,
        #                                     name='sum_logdet_jacobian')
        sum_logdet_jacobian = logdet_jacobian
        print "sum_logdet_jacobian shape: ", sum_logdet_jacobian.get_shape()
        # First term - Third term
        log_qk_zk = log_q0_z0 - sum_logdet_jacobian
        print "log_qk_zk shape:", log_qk_zk.get_shape()

        # First component of the second term: p(x|z_k)
        if beta:
            beta_t = tf.minimum(1.0, 0.01 + tf.cast(global_step / 10000, tf.float32))  # global_step
            print "beta_t shape:", beta_t.get_shape()
            print "recons_error_func(prediction, actual) shape:", recons_error_func(prediction, actual).get_shape()
            log_p_x_given_zk = beta_t * recons_error_func(prediction, actual)
            print "log_p_x_given_zk shape:", log_p_x_given_zk.get_shape()
            log_p_zk = beta_t * tf.reduce_sum(gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu)))
            print "log_p_zk shape:", log_p_zk.get_shape()
        else:
            log_p_x_given_zk = recons_error_func(prediction, actual)
            print "log_p_x_given_zk shape:", log_p_x_given_zk.get_shape()
            log_p_zk = tf.reduce_sum(gaussian_log_pdf(zk, tf.zeros_like(mu), tf.ones_like(mu)))
            print "log_p_zk shape:", log_p_zk.get_shape()

        recons_loss = log_p_x_given_zk
        kl_loss = log_qk_zk - log_p_zk
        _elbo_loss = tf.reduce_mean(kl_loss + recons_loss, name="elbo_loss")

        # Summary
        # tf.summary.scalar("reconstruction_loss", reconstr_loss)
        # tf.summary.scalar("latent_loss", latent_loss)
        tf.summary.scalar("elbo_loss", _elbo_loss)
        merged_summary_losses = tf.summary.merge_all()
        return (recons_loss, kl_loss, _elbo_loss), merged_summary_losses


def mse_vanilla_vae_loss(x, x_reconstr, z_mu, z_var):
    """
    Uses mean squared error as a reconstruction loss instead of the Bernoulli loss. Returns mean of the reconstruction
    KL loss.

    :param x: Actual data
    :param x_reconstr: Reconstructed data
    :param z_mu: Mean of the distribution from which we do the sampling for, for reconstruction.
    :param z_var: Standard deviation of the distribution from which we do the sampling for, for reconstruction.

    :return: Mean of the two error terms.
    """

    print "x shape:", x.get_shape()
    print "x_reconstr:", x_reconstr.get_shape()
    print "z_mu:", z_mu.get_shape()
    print "z_var:", z_var.get_shape()
    reconstr_loss = tf.reduce_sum((x - x_reconstr) ** 2, name="reconstruction_loss")
    print "reconstr loss:", reconstr_loss.get_shape()
    # latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(1e-6 + z_var) - tf.square(z_mu) - \
    #                      tf.exp(tf.log(1e-6 + z_var)), axis = None)
    latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(1e-6 + z_var) - tf.square(z_mu) - z_var,
                                       axis=None, name="latent_loss")
    print "latent loss:", latent_loss.get_shape()
    loss = tf.reduce_mean(reconstr_loss + latent_loss, name="recons_plus_latent_loss")

    # Summary
    tf.summary.scalar("reconstruction_loss", reconstr_loss)
    tf.summary.scalar("latent_loss", latent_loss)
    tf.summary.scalar("recons_plus_latent_loss", loss)
    merged_summary_losses = tf.summary.merge_all()

    return loss, merged_summary_losses


def vanilla_vae_loss(x, x_reconstr, z_mu, z_var):
    print "x shape:", x.get_shape()
    print "x_reconstr:", x_reconstr.get_shape()
    print "z_mu:", z_mu.get_shape()
    print "z_var:", z_var.get_shape()
    reconstr_loss = -tf.reduce_sum(x * tf.log(1e-6 + x_reconstr) + (1 - x) * tf.log(1e-6 + 1 - x_reconstr),
                                   axis=None)
    print "reconstr loss:", reconstr_loss.get_shape()
    # latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(1e-6 + z_var) - tf.square(z_mu) - \
    #                      tf.exp(tf.log(1e-6 + z_var)), axis = None)
    latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(1e-6 + z_var) - tf.square(z_mu) - z_var,
                                       axis=None)
    print "latent loss:", latent_loss.get_shape()
    loss = tf.reduce_mean(reconstr_loss + latent_loss)
    return loss