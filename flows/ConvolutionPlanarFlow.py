import numpy as np
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
    def tf_rot180(filtr, ax=[0]):
        """
        Rotate filter by 180 degrees.
        """
        return tf.reverse(filtr, axis=ax)

    @staticmethod
    def conv1d(z, W, strides=1):
        # Conv1D wrapper
        temp = tf.nn.conv1d(z, W, stride=strides, padding="SAME")
        return temp

    def convolution_planar_flow(self, z, flow_params, num_flows, n_latent_dim, filter_width=3):
        """
        z shape: (bs,ts,?); us, bs shape: (ts, bs, ?); ws shape: (1, num_flows * filter_width * n_latent_dim * 1)
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
                          ws[:, k * filter_width * n_latent_dim * 1:(k + 1) * filter_width * n_latent_dim * 1], \
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
                w = tf.reshape(w, shape=[filter_width, n_latent_dim, 1])  # w shape: (1, 6); reshaped w shape: (3, 2, 1)
                print "reshaped w shape:", w.get_shape()

                # Step 1: Flip the filter by 180 degrees and then convolve. If filter isn't rotated, then tf.nn.conv2d,
                # which is actually called by tf.nn.conv1d, performs a correlation rather than convolution.
                # z shape: (bs, ts, ?), w shape: (3, 2, 1)
                wz = self.conv1d(z, w)  # wz shape: (bs, ts, 1)
                print "wz shape:", wz.get_shape()
                print "b with dims expanded:", tf.expand_dims(b, 2)

                # Step 2:
                # Extend dimension of b. Original b shape: (bs, ts). After expanding dimensions it is (bs, ts, 1).
                wzb = wz + tf.expand_dims(b, 2)  # wzb shape: (bs, ts, 1)
                print "wzb shape:", wzb.get_shape()

                # Step 3: wzb should have a shape of (20,) as tanh takes a real number as input.
                print "tanh(wzb) shape:", tf.nn.tanh(wzb).get_shape()
                z = z + u * tf.nn.tanh(wzb, name="tanh_wzb")  # transformed z: (bs,ts,?)
                print "transformed z shape:", z.get_shape()
                print "dtanh wzb shape:", self.dtanh(wzb).get_shape()  # (bs, ts, 1)

                # Step 4: First transpose the filter to match the dimensions of psi. Then rotate it for convolution.
                # psi shape: (bs, ts, 2)
                psi = self.conv1d(self.dtanh(wzb), self.tf_rot180(tf.transpose(w, perm=[0, 2, 1])))
                print "psi shape:", psi.get_shape()

                # Step 5:
                # u shape: (bs, ts, ?), psi shape: (bs, ts, 2), psi_ut:(bs, ts, 2)
                # psi_ut = tf.multiply(psi, u, name="psi_ut")
                psi_ut = tf.transpose(tf.matmul(tf.transpose(u, perm=[0, 2, 1]), psi, name="psi_u"), perm=[0, 2, 1],
                                      name="psi_ut")  # Reshaped from (bs, ?, 1) to (bs, 1, ?).
                print "psi_ut shape:", psi_ut.get_shape()

                # Step 6:
                logdet_jacobian = tf.log(tf.abs(1 + psi_ut) + 1e-10)  # logdet_jacobian shape: (bs, ts, 2)
                # singular_value = tf.stop_gradient(tf.svd(psi_ut, compute_uv=False))
                # tf.stop_gradient(singular_value, name="stop_gradient_svd")
                # singular_value = np.linalg.svd(psi_ut, compute_uv=0)
                # logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10), axis=1)
                print "logdet_jacobian shape:", logdet_jacobian.get_shape()
                # _logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10))
                # log_detjs.append(tf.expand_dims(logdet_jacobian, axis=1))
                log_detjs.append(logdet_jacobian)
                print "shape of first element in list:", log_detjs[0].get_shape()
            # Concatenated vertically. Below, logdet_jacobian shape: (bs, numFlows*ts, 2).
            logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)
            print "logdet_jacobian inside Normalizing Planar flow:", logdet_jacobian.get_shape()
            sum_logdet_jacobian = tf.reduce_sum(logdet_jacobian)  # shape: ()
            print "sum_logdet_jacobian inside Normalizing Planar flow:", sum_logdet_jacobian.get_shape()
        return tf.transpose(z, perm=[1, 0, 2]), sum_logdet_jacobian



