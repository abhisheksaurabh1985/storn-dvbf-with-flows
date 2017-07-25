import numpy as np


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
