import numpy as np
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


def rollout(env, n_samples, n_timesteps, n_environment_steps=None, split_batch_ratio=1, learned_reward=True,
            filter_mode=False, z_0=None, fn_action=None, render=False):
    """
    Authored by Max. 
    """
    X = []
    U = []
    if filter_mode:
        Z = []
    if render:
        IMG = []

    if fn_action == None:
        fn_action = env.action_space.sample

    if filter_mode:
        one_step = fn_action

    if n_environment_steps == None:
        n_environment_steps = n_timesteps
    n_samples = n_samples // split_batch_ratio

    for i in range(n_samples):
        U.append([])
        X.append([])
        if filter_mode:
            Z.append([z_0])
        if render:
            IMG.append([])

        obs = env.reset()
        reward = np.random.randn()
        if learned_reward:
            obs = np.concatenate([obs.ravel(), [reward]])

        if filter_mode:
            actions = env.action_space.sample()
        else:
            actions = fn_action()

        for j in range(n_environment_steps):
            U[-1].append(actions)
            X[-1].append(obs)
            if render:
                IMG[-1].append(env.render(mode='rgb_array'))
            obs, reward, done, info = env.step(actions)
            if learned_reward:
                obs = np.concatenate([obs.ravel(), [reward]])

            if filter_mode:
                latent, actions = one_step(Z[-1][-1], X[-1][-1], U[-1][-1])
                Z[-1].append(latent)
            else:
                actions = fn_action()

    X = np.array(X)
    U = np.array(U)

    X = X.reshape((X.shape[0], X.shape[1], -1))
    U = U.reshape((U.shape[0], U.shape[1], -1))

    X = X.swapaxes(0, 1)
    U = U.swapaxes(0, 1)

    if n_environment_steps != n_timesteps:
        batch_index = np.repeat(np.arange(n_samples), split_batch_ratio)
        big_X = X[:, batch_index, :]
        big_U = U[:, batch_index, :]

        index = np.random.randint(0, n_environment_steps - n_timesteps, size=(n_samples * split_batch_ratio,))

        X = np.concatenate([big_X[s:n_timesteps + s, i:i + 1, :] for i, s in enumerate(index)], 1)
        U = np.concatenate([big_U[s:n_timesteps + s, i:i + 1, :] for i, s in enumerate(index)], 1)

    if render:
        IMG = np.array(IMG)
        IMG = IMG.swapaxes(0, 1)
        return X, U, IMG
    else:
        return X, U
