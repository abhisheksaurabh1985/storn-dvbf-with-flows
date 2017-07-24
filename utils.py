import numpy as np
from matplotlib import collections as mc
import matplotlib.animation as animation
from pylab import *
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import matplotlib.pylab as plt
from IPython import display


class ActionSpaceWrapper():
    def __init__(self, action_space, n_steps=1):
        self.action_space = action_space
        self.n_steps = n_steps

    def sample(self):
        return np.concatenate([self.action_space.sample()
                               for i in range(self.n_steps)])


class EnvWrapper():
    def __init__(self, env, n_steps=1):
        self.env = env
        self.n_steps = n_steps
        self.action_space = ActionSpaceWrapper(env.action_space, n_steps)

    def step(self, actions):
        n_control = len(actions) // self.n_steps
        for i in range(self.n_steps):
            out = self.env.step(actions[i * n_control: (i + 1) * n_control])
        return out

    def render(self, mode):
        return self.env.render(mode=mode)

    def reset(self):
        return self.env.reset()

# Stuff below taken from http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)

IMG_TAG = """<img src="data:image/gif;base64,{0}" alt="some_text">"""

def anim_to_gif(anim):
    data="0"
    with NamedTemporaryFile(suffix='.gif') as f:
        anim.save(f.name, writer='imagemagick', fps=20);
        data = open(f.name, "rb").read()
        data = data.encode("base64")
    return IMG_TAG.format(data)

def display_animation(anim,gif=False):
    plt.close(anim._fig)
    if gif:
        return HTML(anim_to_gif(anim))
    else:
        return HTML(anim_to_html(anim))


def mean_hexbin(XY,Z,s=57,gridsize=20,alpha=1.0, colormap='jet'):
    hexdata = plt.hexbin(XY[:,0], XY[:,1], gridsize=gridsize)
    plt.close()

    plt.figure(figsize=(20, 13))

    counts = hexdata.get_array()
    verts = hexdata.get_offsets()

    points = XY.reshape(XY.shape[0], -1)

    binindex = np.argmin( np.concatenate([((p-verts) ** 2).sum(1)[np.newaxis, :] for p in points], 0), 1)
    print(binindex.shape)
    color = [np.mean(np.array(Z)[binindex == i]) for i in xrange(verts.shape[0])]
    _ = plt.scatter(verts[:,0], verts[:,1],
                    c=color, s=s, edgecolor='',
                    alpha=alpha, marker='h', cmap=colormap)
    plt.colorbar(_)


def rollout(env, n_samples, n_timesteps, n_environment_steps=None, split_batch_ratio=1, learned_reward=True, filter_mode=False, z_0=None, fn_action=None, render=False):
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

        X = np.concatenate([big_X[s:n_timesteps + s, i:i+1, :] for i,s in enumerate(index)], 1)
        U = np.concatenate([big_U[s:n_timesteps + s, i:i+1, :] for i,s in enumerate(index)], 1)


    if render:
        IMG = np.array(IMG)
        IMG = IMG.swapaxes(0, 1)
        return X, U, IMG
    else:
        return X, U


def rollout_multiagent(env, n_agents, n_samples, n_timesteps, learned_reward=True, filter_mode=False, z_0=None, fn_action=None, render=False):
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
            
        for j in range(n_timesteps):
            U[-1].append(actions)
            X[-1].append(obs)
            if render:
                IMG[-1].append(env.render(mode='rgb_array'))
            obs, reward, done, info = env.step(actions)
            if learned_reward:
                obs = np.concatenate([obs.ravel(), [reward]])
                
            if filter_mode:
                latent = []
                actions = []
                n_obs = X[-1][-1].shape[0] // n_agents
                n_control = U[-1][-1].shape[0] // n_agents
                n_latent = Z[-1][-1].shape[0] // n_agents
                for a in range(n_agents):
                    _latent, _actions = one_step(
                        Z[-1][-1][a * n_latent:(a + 1) * n_latent],
                        X[-1][-1][a * n_obs:(a + 1) * n_obs],
                        U[-1][-1][a * n_control:(a + 1) * n_control])
                    latent.append(_latent)
                    actions.append(_actions)
                latent = np.concatenate(latent)
                actions = np.concatenate(actions)
                Z[-1].append(latent)
            else:
                actions = fn_action()

    X = np.array(X)
    U = np.array(U)
        
    X = X.reshape((X.shape[0], X.shape[1], -1))
    U = U.reshape((U.shape[0], U.shape[1], -1))

    X = X.swapaxes(0, 1)
    U = U.swapaxes(0, 1)

    X = np.concatenate([X[:, :, a * (X.shape[2] // n_agents):(a + 1) * (X.shape[2] // n_agents)] for a in range(n_agents)], 1)
    U = np.concatenate([U[:, :, a * (U.shape[2] // n_agents):(a + 1) * (U.shape[2] // n_agents)] for a in range(n_agents)], 1)

    if render:
        IMG = np.array(IMG)
        IMG = IMG.swapaxes(0, 1)
        return X, U, IMG
    else:
        return X, U


def waterfilling(emp, P=1.0):
    def _waterfill(e, P):
        def h(s, p):
            return 0.5 * np.log(1.0 + s * p)
        step = 0.1
        _P = P
        ind = np.argsort(-e)
        i = 0
        out = 0.0
        p = 0
        while _P > 0 and i < (len(e) - 1):
            if h(e[ind[i]], p + step)-h(e[ind[i]], p) > h(e[ind[i+1]], step)-h(e[ind[i+1]], 0):
                p += step
                _P -= step
                if _P <= 0:
                    out += h(e[ind[i]], p)
            else:
                out += h(e[ind[i]], p)
                p = 0
                i += 1
        if _P > 0:
            out += h(e[ind[i]], _P)
        return out
    return [_waterfill(e, P=P) for e in emp]


def empowerment(self, Z):
    Ts = self.sess.run(self.empowerment_T, feed_dict={self.empowerment_z: Z.reshape((-1, self.n_latent))}) 
    Cs = self.sess.run(self.empowerment_C, feed_dict={self.empowerment_z: Z.reshape((-1, self.n_latent))})

    def emp(T, C):
        if C==None:
            _, s, _ = np.linalg.svd(T)
            return s
        U, state_s, _ = np.linalg.svd(np.dot(C.T, C))
        _, s, _ = np.linalg.svd(np.dot(np.diag(1.0/np.sqrt(state_s)), np.dot(U.T,T.T)))
        return s

    return np.array(waterfilling(np.array([emp(T, C) for T, C in zip(Ts, Cs)])))


def plot_manif(ax, LM, LV, C, plot_scatter=True, plot_lines=True, axes=(0, 1),
               colormap='jet', alpha=.1, lw=1, s=50, sample=True):
    n_time_steps = LM.shape[0]
    idxs = range(LM.shape[1])
    a, b = axes

    if sample:
        L = LM + (LV ** .5) * np.random.standard_normal(LV.shape)
    else:
        L = LM

    if plot_lines:
        lines = []
        for i in idxs:
            lines += [((LM[t, i, a], LM[t, i, b]),
                      (LM[t + 1, i, a], LM[t + 1, i, b]))
                      for t in range(n_time_steps - 1)]
        lc = mc.LineCollection(lines, alpha=.1, colors='k', zorder=2)
        ax.add_collection(lc)

    if plot_scatter:
        for i in idxs:
            ax.scatter(L[:, i, a], L[:, i, b], c=C[:, i], s=s, alpha=alpha, lw=lw, cmap=colormap, zorder=2)


def ani_frame(video, cmap='binary'):
    dpi = 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(video[0], cmap=cmap, interpolation='nearest')
    #im.set_clim([-127,128])
    fig.set_size_inches([5,5])

    tight_layout()

    def update_img(n):
        tmp = video[n]
        im.set_data(tmp)
        return im

    ani = animation.FuncAnimation(
        fig,update_img, video.shape[0], interval=30)
    return ani


# Stuff below taken from http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
            video = open(f.name, "rb").read()
        anim._encoded_video = video.encode("base64")

    return VIDEO_TAG.format(anim._encoded_video)


IMG_TAG = """<img src="data:image/gif;base64,{0}" alt="some_text">"""


def anim_to_gif(anim):
    data="0"
    with NamedTemporaryFile(suffix='.gif') as f:
        anim.save(f.name, writer='imagemagick', fps=20);
        data = open(f.name, "rb").read()
        data = data.encode("base64")
    return IMG_TAG.format(data)


def display_animation(anim,gif=False):
    plt.close(anim._fig)
    if gif:
        return HTML(anim_to_gif(anim))
    else:
        return HTML(anim_to_html(anim))


def animate(frames):
    frames=np.asarray(frames)
    
    idx = 0
    ani = ani_frame(frames[:,:,:,:])
    plt.close(ani._fig)
    return anim_to_gif(ani)
