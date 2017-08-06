import nn_utilities

import tensorflow as tf


class STORN(object):
    def __init__(self, data_dim, time_steps, n_hidden_units_enc, n_hidden_units_dec, n_latent_dim, batch_size,
                 learning_rate=0.001, flow_type="NoFlow", num_flows=None, mu_init=0, sigma_init=0.00001,
                 decoder_output_function=tf.identity, activation_function=tf.nn.relu):
        self.data_dim = data_dim
        self.time_steps = time_steps
        self.n_hidden_units_enc = n_hidden_units_enc
        self.n_hidden_units_dec = n_hidden_units_dec
        self.n_latent_dim = n_latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.flow_type = flow_type
        # self.nf_planar = nf_planar  # Boolean for planar normalizing flows
        self.num_flows = num_flows  # Number of times flow will be applied
        self.decoder_output_function = decoder_output_function
        self.activation_function = activation_function

        # Initializers for encoder parameters
        self.init_wxhe = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                       self.data_dim,
                                                                       self.mu_init, self.sigma_init)
        self.init_whhe = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                       self.n_hidden_units_enc,
                                                                       self.mu_init, self.sigma_init)
        self.init_bhe = nn_utilities.initialize_bias_with_zeros(self.n_hidden_units_enc)
        self.init_whmu = nn_utilities.initialize_weights_random_normal(self.n_latent_dim,
                                                                       self.n_hidden_units_enc,
                                                                       self.mu_init, self.sigma_init)
        self.init_bhmu = nn_utilities.initialize_bias_with_zeros(self.n_latent_dim)
        self.init_whsigma = nn_utilities.initialize_weights_random_normal(self.n_latent_dim,
                                                                          self.n_hidden_units_enc,
                                                                          self.mu_init,
                                                                          self.sigma_init)
        self.init_bhsigma = nn_utilities.initialize_bias_with_zeros(self.n_latent_dim)

        # Initializers for the decoder parameters
        self.init_dec_wzh = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_dec,
                                                                          self.n_latent_dim,
                                                                          self.mu_init,
                                                                          self.sigma_init)
        self.init_dec_bzh = nn_utilities.initialize_bias_with_zeros(self.n_hidden_units_dec)
        self.init_dec_whhd = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_dec,
                                                                           self.n_hidden_units_dec,
                                                                           self.mu_init,
                                                                           self.sigma_init)
        self.init_dec_wxhd = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_dec, self.data_dim,
                                                                           self.mu_init,
                                                                           self.sigma_init)
        self.init_dec_bhd = nn_utilities.initialize_bias_with_zeros(self.n_hidden_units_dec)
        self.init_dec_whx = nn_utilities.initialize_weights_random_normal(self.data_dim, self.n_hidden_units_dec,
                                                                          self.mu_init,
                                                                          self.sigma_init)
        self.init_dec_bhx = nn_utilities.initialize_bias_with_zeros(self.data_dim)

        self.init_dec_w_mu = nn_utilities.initialize_weights_random_normal(self.data_dim, self.n_hidden_units_dec,
                                                                          self.mu_init,
                                                                          self.sigma_init)
        self.init_dec_w_var = nn_utilities.initialize_weights_random_normal(self.data_dim, self.n_hidden_units_dec,
                                                                          self.mu_init,
                                                                          self.sigma_init)
        self.init_dec_b_h_mu = nn_utilities.initialize_bias_with_zeros(self.data_dim)
        self.init_dec_b_h_var = nn_utilities.initialize_bias_with_zeros(self.data_dim)

        # Initializers for planar normalizing flow parameters in the encoder
        if self.flow_type == "Planar":
            self.init_w_us = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                           self.num_flows * self.n_latent_dim,
                                                                           self.mu_init,
                                                                           self.sigma_init)
            self.init_b_us = nn_utilities.initialize_bias_with_zeros(self.num_flows * self.n_latent_dim)
            self.init_w_ws = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                           self.num_flows * self.n_latent_dim,
                                                                           self.mu_init,
                                                                           self.sigma_init)
            self.init_b_ws = nn_utilities.initialize_bias_with_zeros(self.num_flows * self.n_latent_dim)
            self.init_w_bs = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                           self.num_flows * self.n_latent_dim,
                                                                           self.mu_init,
                                                                           self.sigma_init)
            self.init_b_bs = nn_utilities.initialize_bias_with_zeros(self.num_flows * self.n_latent_dim)

        # Initializers for radial normalizing flow parameters in the encoder
        if self.flow_type == "Radial":
            self.init_w_z0s = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                            self.num_flows * self.n_latent_dim,
                                                                            self.mu_init,
                                                                            self.sigma_init)
            self.init_b_z0s = nn_utilities.initialize_bias_with_zeros(self.num_flows * self.n_latent_dim)
            self.init_w_alphas = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                               self.num_flows,
                                                                               self.mu_init,
                                                                               self.sigma_init)
            self.init_b_alphas = nn_utilities.initialize_bias_with_zeros(self.num_flows)
            self.init_w_betas = nn_utilities.initialize_weights_random_normal(self.n_hidden_units_enc,
                                                                              self.num_flows,
                                                                              self.mu_init,
                                                                              self.sigma_init)
            self.init_b_betas = nn_utilities.initialize_bias_with_zeros(self.num_flows)

    def encoding_step(self, h_t, x_t):
        """
        The encoder has one set of recurrent connection. State h_{t+1} is calculated based on the previous
        recurrent state h_t and current input x_{t+1}.

        This function is called by the tf.scan function inside encoder_rnn.

        :param h_t: Refers to the previous output. At the start this has a shape (mb_size, n_hidden_units_enc).
        :param x_t: Refers to the current_input. At the start is has the shape (?,5).

        :return output_encoding_step: Output at the end of each run. Has a shape (6,100).
        """
        print "self W_xhe shape:", self.W_xhe.get_shape()
        print "self W_hhe shape:", self.W_hhe.get_shape()
        print "h_t shape:", h_t.get_shape()
        print "x_t shape:", x_t.get_shape()
        print "b_he shape:", self.b_he.get_shape()
        first_term = tf.tensordot(self.W_xhe, tf.cast(x_t, tf.float32), axes=[[1], [1]], name="enc_first_term")
        print "first_term shape", first_term.get_shape()
        second_term = tf.tensordot(self.W_hhe, tf.cast(h_t, tf.float32), axes=[[1], [1]], name="enc_second_term")
        print "second_term shape", second_term.get_shape()
        output_encoding_step = tf.transpose(self.activation_function(first_term + second_term + self.b_he),  name="output_encoding_step")
        print "output_encoding_shape", output_encoding_step.get_shape()
        return output_encoding_step

    def reparametrize_z(self, z_mu, z_var):
        """
        Sampling from a normal distribution with the mean and sigma given.

        :param z_mu: Mean of the distribution for each item in the mini-batch for each time step.
                     Has a shape (T,B,D) where T, B and D refer to time step, batch size and dimension respectively.
        :param z_var: Standard deviation for each item in the mini-batch for each time step. Dimension
                      same as that of z_mu.
        :return: self.z: Sampled (aka reparametrized) z with shape (T,B,D).
        """
        eps = tf.random_normal(shape=tf.shape(z_mu), mean=0, stddev=1)  # Shape: (100,6,2)
        print "eps shape", eps.get_shape()
        self.z = tf.add(z_mu, tf.multiply(tf.sqrt(z_var), eps))  # Shape: (100,6,2)
        print "z shape", self.z.get_shape()
        return self.z

    def encoder_rnn(self, x):
        """
        RNN as an encoder network in STORN. For a given input x it returns a compressed representation
        in the latent space.

        :param x: Input time series data with dimension (T,B,D).

        :return self.mu_encoder: Mean of the data for each item in the batch at each time step. Has a shape (T,B,D).
        :return self.log_sigma_encoder: Standard deviation of the data for each item in the batch at each time step. Has
         a shape (T,B,D).
        """
        with tf.variable_scope('encoder_rnn'):
            self.W_xhe = tf.Variable(initial_value=self.init_wxhe, name="W_xhe", dtype=tf.float32)
            self.W_hhe = tf.Variable(initial_value=self.init_whhe, name="W_hhe", dtype=tf.float32)
            self.b_he = tf.Variable(initial_value=self.init_bhe, name="b_he", dtype=tf.float32)
            self.W_hmu = tf.Variable(initial_value=self.init_whmu, name="W_hmu", dtype=tf.float32)
            self.b_hmu = tf.Variable(initial_value=self.init_bhmu, name="b_hmu", dtype=tf.float32)
            self.W_hsigma = tf.Variable(initial_value=self.init_whsigma, name="W_hsigma", dtype=tf.float32)
            self.b_hsigma = tf.Variable(initial_value=self.init_bhsigma, name="b_hsigma", dtype=tf.float32)

            if self.flow_type == "Planar":
                print "self.flow_type in encoder_rnn", self.flow_type
                with tf.variable_scope("encoder_nf_planar"):
                    self.W_us = tf.Variable(initial_value=self.init_w_us, name="W_us", dtype=tf.float32)
                    self.b_us = tf.Variable(initial_value=self.init_b_us, name="b_us", dtype=tf.float32)
                    self.W_ws = tf.Variable(initial_value=self.init_w_ws, name="W_ws", dtype=tf.float32)
                    self.b_ws = tf.Variable(initial_value=self.init_b_ws, name="b_ws", dtype=tf.float32)
                    self.W_bs = tf.Variable(initial_value=self.init_w_bs, name="W_bs", dtype=tf.float32)
                    self.b_bs = tf.Variable(initial_value=self.init_b_bs, name="b_bs", dtype=tf.float32)
            elif self.flow_type == "Radial":
                print "self.flow_type in encoder_rnn", self.flow_type
                with tf.variable_scope("encoder_nf_radial"):
                    self.W_z0s = tf.Variable(initial_value=self.init_w_z0s, name="W_z0s", dtype=tf.float32)
                    self.b_z0s = tf.Variable(initial_value=self.init_b_z0s, name="b_z0s", dtype=tf.float32)
                    self.W_alphas = tf.Variable(initial_value=self.init_w_alphas, name="W_alphas", dtype=tf.float32)
                    self.b_alphas = tf.Variable(initial_value=self.init_b_alphas, name="b_alphas", dtype=tf.float32)
                    self.W_betas = tf.Variable(initial_value=self.init_w_betas, name="W_betas", dtype=tf.float32)
                    self.b_betas = tf.Variable(initial_value=self.init_b_betas, name="b_betas", dtype=tf.float32)

            # Number of time steps
            states_0 = tf.zeros([tf.shape(x)[1], self.n_hidden_units_enc], tf.float32, name="enc_states_0")
            print "states_0 shape", states_0.get_shape()
            print "x shape", x.get_shape()  # (100, ?, 5)
            states = tf.scan(self.encoding_step, x, initializer=states_0, name='states')
            print "states shape", states.get_shape()  # shape:(timeSteps,miniBatchSize,nUnitsEncoder)

            # Reshape states
            _states = tf.reshape(states, [-1, self.n_hidden_units_enc],
                                 name="encoder_states")  # Shape:(timeSteps * miniBatchSize, nUnitsEncoder)
            print "_states shape", _states.get_shape()
            print "W_hmu shape", self.W_hmu.get_shape()
            print "b_hmu shape", self.b_hmu.get_shape()

            if self.flow_type == "Planar":
                # Parameters of the distribution
                self.mu_encoder = tf.tensordot(self.W_hmu, _states, axes=[[1], [1]], name="mu_encoder")
                print "mu_encoder shape", self.mu_encoder.get_shape()  # Shape:(z_dim, timeSteps*miniBatchSize)
                self.mu_encoder = tf.reshape(tf.transpose(self.mu_encoder),
                                             (self.time_steps, self.batch_size, -1),
                                             name="reshaped_mu_encoder")  # Shape:(timeSteps, miniBatchSize, z_dim)
                print "mu_encoder 3D shape", self.mu_encoder.get_shape()

                print "self.W_hsigma shape inside encoder rnn:", self.W_hsigma.get_shape()
                print "_states shape inside encoder rnn:", _states.get_shape()
                self.log_sigma_encoder = tf.tensordot(self.W_hsigma, _states,
                                                      axes=[[1], [1]],
                                                      name="log_sigma_encoder")  # Shape:(z_dim,timeSteps*miniBatchSize)
                print "########"
                print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
                print "########"
                self.log_sigma_encoder = tf.reshape(tf.transpose(self.log_sigma_encoder),
                                                    (self.time_steps, self.batch_size, -1),
                                                    name="reshaped_log_sigma_encoder")  # Shape:(timeSteps,miniBatchSize,z_dim)
                print "########"
                print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
                print "########"

                # Planar normalizing flow parameters
                print "self.W_us in encoder rnn:", self.W_us.get_shape()
                print "_states in encoder rnn:", _states.get_shape()
                self.us = tf.tensordot(self.W_us, _states, axes=[[0], [1]], name="planar_nf_us")
                self.us = tf.reshape(tf.transpose(self.us),
                                     (self.time_steps, self.batch_size, -1),
                                     name="reshaped_planar_nf_us")  # Shape:(timeSteps, miniBatchSize, z_dim)

                self.ws = tf.tensordot(self.W_ws, _states, axes=[[0], [1]], name="planar_nf_ws")
                self.ws = tf.reshape(tf.transpose(self.ws),
                                     (self.time_steps, self.batch_size, -1),
                                     name="reshaped_planar_nf_ws")  # Shape:(timeSteps, miniBatchSize, z_dim)

                self.bs = tf.tensordot(self.W_bs, _states, axes=[[0], [1]], name="planar_nf_bs")
                self.bs = tf.reshape(tf.transpose(self.bs),
                                     (self.time_steps, self.batch_size, -1),
                                     name="reshaped_planar_nf_bs")  # Shape:(timeSteps, miniBatchSize, z_dim)

                planar_flow_params = (self.us, self.ws, self.bs)
                return self.mu_encoder, self.log_sigma_encoder, planar_flow_params
            elif self.flow_type == "Radial":
                # Parameters of the distribution
                self.mu_encoder = tf.tensordot(self.W_hmu, _states, axes=[[1], [1]], name="mu_encoder")
                print "mu_encoder shape", self.mu_encoder.get_shape()  # Shape:(z_dim, timeSteps*miniBatchSize)
                self.mu_encoder = tf.reshape(tf.transpose(self.mu_encoder),
                                             (self.time_steps, self.batch_size, -1),
                                             name="reshaped_mu_encoder")  # Shape:(timeSteps, miniBatchSize, z_dim)
                print "mu_encoder 3D shape", self.mu_encoder.get_shape()

                print "self.W_hsigma shape inside encoder rnn:", self.W_hsigma.get_shape()
                print "_states shape inside encoder rnn:", _states.get_shape()
                self.log_sigma_encoder = tf.tensordot(self.W_hsigma, _states,
                                                      axes=[[1], [1]],
                                                      name="log_sigma_encoder")  # Shape:(z_dim,timeSteps*miniBatchSize)
                print "########"
                print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
                print "########"
                self.log_sigma_encoder = tf.reshape(tf.transpose(self.log_sigma_encoder),
                                                    (self.time_steps, self.batch_size, -1),
                                                    name="reshaped_log_sigma_encoder")  # Shape:(timeSteps,miniBatchSize,z_dim)
                print "########"
                print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
                print "########"

                # Radial normalizing flow parameters
                print "self.W_z0s in encoder rnn:", self.W_z0s.get_shape()
                print "_states in encoder rnn:", _states.get_shape()
                self.z0s = tf.tensordot(self.W_z0s, _states, axes=[[0], [1]], name="radial_nf_z0s")
                self.z0s = tf.reshape(tf.transpose(self.z0s),
                                      (self.time_steps, self.batch_size, -1),
                                      name="reshaped_radial_nf_z0s")  # Shape:(timeSteps, miniBatchSize, z_dim)

                self.alphas = tf.tensordot(self.W_alphas, _states, axes=[[0], [1]], name="radial_nf_alphas")
                self.alphas = tf.reshape(tf.transpose(self.alphas),
                                         (self.time_steps, self.batch_size, -1),
                                         name="reshaped_radial_nf_alphas")  # Shape:(timeSteps, miniBatchSize, z_dim)

                self.betas = tf.tensordot(self.W_betas, _states, axes=[[0], [1]], name="radial_nf_betas")
                self.betas = tf.reshape(tf.transpose(self.betas),
                                        (self.time_steps, self.batch_size, -1),
                                        name="reshaped_radial_nf_betas")  # Shape:(timeSteps, miniBatchSize, z_dim)
                radial_flow_params = (self.z0s, self.alphas, self.betas)
                return self.mu_encoder, self.log_sigma_encoder, radial_flow_params
            elif self.flow_type == "NoFlow":
                # Parameters of the distribution
                self.mu_encoder = tf.tensordot(self.W_hmu, _states, axes=[[1], [1]], name="mu_encoder")
                print "mu_encoder shape", self.mu_encoder.get_shape()  # Shape:(z_dim, timeSteps*miniBatchSize)
                self.mu_encoder = tf.reshape(tf.transpose(self.mu_encoder),
                                             (self.time_steps, self.batch_size, -1),
                                             name="reshaped_mu_encoder")  # Shape:(timeSteps, miniBatchSize, z_dim)
                print "mu_encoder 3D shape", self.mu_encoder.get_shape()
                self.log_sigma_encoder = tf.tensordot(self.W_hsigma, _states,
                                                      axes=[[1], [1]],
                                                      name="log_sigma_encoder")  # Shape:(z_dim, timeSteps*miniBatchSize)
                print "########"
                print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
                print "########"
                self.log_sigma_encoder = tf.reshape(tf.transpose(self.log_sigma_encoder),
                                                    (self.time_steps, self.batch_size, -1),
                                                    name="reshaped_log_sigma_encoder")  # Shape:(timeSteps, miniBatchSize, z_dim)
                print "########"
                print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
                print "########"
                flow_params = None
                return self.mu_encoder, self.log_sigma_encoder, flow_params

    def decoding_step(self, previous_output, z_t):
        """
        Returns the recurrent state at the previous time step and the reconstruction from the data in the latent space.
        Executed by the tf.scan function in decoder_rnn for as many times as there are time steps (eqv to the first
        dimension of z when the function call is made). Data at each time step is used one at a time. Therefore, the
        dimension of z_t is (mini_batch,nDimLatentSpace).

        First input is the previous output, initialized at the time of function call. Second is the current input i.e.
        input in the latent space which is to be reconstructed.

        :param previous_output: Recurrent states and the data to be reconstructed are the previous output which will be
        calculated at each time step based on the input at each time step i.e. z_t.
        :param z_t: Point in the latent space which will be reconstructed. Shape in each iteration of tf.scan is
        (mini_batch, nDimLatentSpace). tf.scan will iterate as many times as there are time steps.
        :return h: Recurrent state outputted at the previous time step i.e. $h_{t-1}$. Has a shape
        (mini_batch, numUnitsDecoder).
        :return x: Reconstruction at each time step. Has a shape (mini_batch, data_dimensions).
        """
        # First element is the initial recurrent state. Second is the initial reconstruction, which isn't needed.
        h_t, _, _, _ = previous_output
        print "Decoding step z_t shape", z_t.get_shape()
        print "Decoding step h_t shape", h_t.get_shape()
        print "Decoding step W_hhd shape", self.W_hhd.get_shape()
        # print "Decoding step W_xhd shape", self.W_xhd.get_shape()
        print "Decoding step b_hd shape", self.b_hd.get_shape()

        # W_hhd:(100,100); h_t:(100,6); W_xhd: (100,5); x_t:(6,5); b_hd:(100,1)
        h = tf.transpose(self.activation_function(tf.tensordot(self.W_hhd, h_t, axes=[[1], [1]], name="dec_rec_first_term_h") +
                                 tf.tensordot(self.W_zh, z_t, axes=[[1], [1]], name="dec_rec_second_term_h") +
                                 self.b_hd), name="decoding_step_tr_h")
        print "Decoding step h shape", h.get_shape()
        print "Decoding step W_hx shape", self.W_hx.get_shape()
        print "Decoding step b_hx shape", self.b_hx.get_shape()
        # x = tf.transpose(tf.identity(tf.tensordot(self.W_hx, h, axes=[[1], [1]]) + self.b_hx), name="x_recons")
        # print "Decoding step x shape", x.get_shape()
        mu_x = tf.transpose(tf.tensordot(self.W_h_mu_dec, h, axes=[[1], [1]]) + self.b_h_mu,
                            name="mu_x_decoder")
        logvar_x = tf.transpose(tf.tensordot(self.W_h_var_dec, h, axes=[[1], [1]]) + self.b_h_var,
                                name="logvar_x_decoder")
        x = self.decoder_output_function(mu_x, name="x_recons")
        return h, x, mu_x, logvar_x

    def decoder_rnn(self, z):
        """
        Returns the input reconstructed from the compressed data obtained from the encoder.

        :param z: Compressed data obtained from the encoder post reparametrization. Has a shape (T,B,D), where D is the
        number of dimensions in the latent space.

        :return self.recons_x: Reconstructed input of shape (T,B,D) where D is the original number of dimensions.
        """
        # Parameters of the decoder network
        with tf.variable_scope('decoder_rnn'):
            self.W_zh = tf.Variable(initial_value=self.init_dec_wzh, name="W_zh", dtype=tf.float32)  # Weights for z_t.
            # self.b_zh = tf.Variable(initial_value=self.init_dec_bzh, name="b_zh", dtype=tf.float32)
            self.W_hhd = tf.Variable(initial_value=self.init_dec_whhd, name="W_hhd", dtype=tf.float32)  # W_rec
            # self.W_xhd = tf.Variable(initial_value=self.init_dec_wxhd, name="W_xhd", dtype=tf.float32)
            self.b_hd = tf.Variable(initial_value=self.init_dec_bhd, name="b_hd", dtype=tf.float32)
            self.W_hx = tf.Variable(initial_value=self.init_dec_whx, name="W_hx", dtype=tf.float32)  # Weights for x_t
            self.b_hx = tf.Variable(initial_value=self.init_dec_bhx, name="b_hx", dtype=tf.float32)

            # Weights for mu and sigma of decoder
            self.W_h_mu_dec = tf.Variable(initial_value=self.init_dec_w_mu, name="W_h_mu_dec", dtype=tf.float32)
            self.W_h_var_dec = tf.Variable(initial_value=self.init_dec_w_var, name="W_h_var_dec", dtype=tf.float32)
            self.b_h_mu = tf.Variable(initial_value=self.init_dec_b_h_mu, name="b_h_mu", dtype=tf.float32)
            self.b_h_var = tf.Variable(initial_value=self.init_dec_b_h_var, name="b_h_var", dtype=tf.float32)

            # Initial recurrent state
            print "z0 first time step shape:", z[0, :, :].get_shape()
            # Compute initial state of the decoding RNN with one set of weights.
            print "z shape decoder_rnn ", z.get_shape()
            print "self.W_zh in decoder_rnn shape", self.W_zh.get_shape()
            # print "self.b_zh in decoder_rnn shape", self.b_zh.get_shape()
            print "reshaped transpose tf.tensordot(self.W_zh, z, axes=[[1],[2]]::", \
                tf.reshape(tf.tensordot(self.W_zh, z, axes=[[1], [2]]),
                           [self.time_steps, -1, self.batch_size]).get_shape()

            # Iterate over each item in the latent space. Initializer will be the first recurrrent state i.e. h0 and the
            # reconstruction at the first time step.
            initial_recurrent_state = tf.random_normal((self.batch_size, self.n_hidden_units_dec),
                                                       mean=0,
                                                       stddev=1,
                                                       name="dec_init_rec_state")
            recons_init_x = tf.random_normal((self.batch_size, self.data_dim),
                                             mean=0,
                                             stddev=1,
                                             name="dec_recons_init_x")
            mu_recons_init_x = tf.random_normal((self.batch_size, self.data_dim),
                                                mean=0,
                                                stddev=1,
                                                name="dec_mu_recons_init_x")
            logvar_recons_init_x = tf.random_normal((self.batch_size, self.data_dim),
                                                    mean=0,
                                                    stddev=1,
                                                    name="dec_logvar_recons_init_x")
            _, self.recons_x, self.mu_recons_x, self.logvar_recons_x = tf.scan(self.decoding_step,
                                                                               z, initializer=(initial_recurrent_state,
                                                                                               recons_init_x,
                                                                                               mu_recons_init_x,
                                                                                               logvar_recons_init_x),
                                                                               name='recons_x')
            print "recons x shape", self.recons_x.get_shape()
            return self.mu_recons_x, self.logvar_recons_x

    def reconstruct(self, sess, x, data):
        return sess.run(self.recons_x, feed_dict={x: data})

    def get_latent(self, sess, x, data):
        return sess.run(self.z, feed_dict={x: data})
