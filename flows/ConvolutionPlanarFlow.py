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

    @staticmethod
    def conv1d(z, W, strides=1):
        # Conv1D wrapper
        temp = tf.nn.conv1d(z, W, stride=strides, padding="SAME")
        return temp

    def convolution_planar_flow(self, z, flow_params, num_flows, n_latent_dim, filter_width=3):
        """
        z, us, bs shape: (100, 20, ?); ws shape: (1, num_flows * filter_width * n_latent_dim * n_latent_dim)
        """
        us, ws, bs = flow_params
        print "us shape:", us.get_shape()
        print "ws shape:", ws.get_shape()
        print "bs shape:", bs.get_shape()
        print "z shape:", z.get_shape()
        log_detjs = []
        if num_flows == 0:
            sum_logdet_jacobian = tf.Variable(0.0, dtype=tf.float32)
        else:
            for k in range(num_flows):
                u, w, b = us[:, :, k * n_latent_dim:(k + 1) * n_latent_dim], \
                          ws[:, k * filter_width * n_latent_dim * n_latent_dim:(k + 1) * filter_width * n_latent_dim * n_latent_dim], \
                          bs[:, :, k]

                # Reshape u and b s.t. batch size is the first dimension. When applicable, n_time_step is the second
                # dimension and n_latent_dim the third.
                print "$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#"
                print "b before transpose:", b.get_shape()
                u = tf.transpose(u, perm=[1, 0, 2])  # u shape: (20, 100, ?)
                b = tf.transpose(b, perm=[1, 0])  # (20, 100)
                print "u shape:", u.get_shape()
                print "w shape:", w.get_shape()
                print "b shape:", b.get_shape()
                w = tf.reshape(w, shape=[filter_width, n_latent_dim, n_latent_dim])  # w shape: (1, 12); reshaped w shape: (3, 2, 2)
                print "reshaped w shape:", w.get_shape()

                # Step 1:
                wz = self.conv1d(z, w)  # wz shape: (20, 100, 2)
                print "wz shape:", wz.get_shape()
                print "b with dims expanded:", tf.expand_dims(b, 2)

                # Step 2:
                # Extend dimension of b
                wzb = wz + tf.expand_dims(b, 2)  # wzb shape: (20, 100, 2)
                print "wzb shape:", wzb.get_shape()

                # Step 3: wzb should have a shape of (20,) as tanh takes a real number as input.
                print "tanh(wzb) shape:", tf.nn.tanh(wzb).get_shape()
                z = z + u * tf.nn.tanh(wzb, name="tanh_wzb")
                print "transformed z shape:", z.get_shape()
                print "derivative wzb shape:", self.dtanh(wzb).get_shape()  # (?, 100, 2); reshaped w shape: (3, 2, 2)

                # Step 4: d(f*g)/dx = f * dg/dx. Here, dg/dx == 1.
                psi = self.dtanh(wzb) * self.conv1d(tf.ones_like(z, name="ones_for_convolving_with_w"), w)
                print "psi shape:", psi.get_shape()

                # Step 5:
                psi_ut = tf.matmul(tf.transpose(u, perm=[0, 2, 1]), psi, name="psi_u")  # psi_u:(?, 2, 2)
                print "psi_u shape:", psi_ut.get_shape()

                # Step 6:
                logdet_jacobian = tf.log(tf.abs(1 + psi_ut) + 1e-10)  # logdet_jacobian shape: (?, 2, 2)
                print "logdet_jacobian shape:", logdet_jacobian.get_shape()
                log_detjs.append(logdet_jacobian)
            logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)  # Concatenated vertically
            print "logdet_jacobian inside Normalizing Planar flow:", logdet_jacobian.get_shape()
            sum_logdet_jacobian = tf.reduce_sum(logdet_jacobian)
            print "sum_logdet_jacobian inside Normalizing Planar flow:", sum_logdet_jacobian.get_shape()
        return tf.transpose(z, perm=[1, 0, 2]), sum_logdet_jacobian



