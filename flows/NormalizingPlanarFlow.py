import tensorflow as tf


class NormalizingPlanarFlow(object):

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
    def tf_norm(x):
        return tf.sqrt(tf.reduce_sum(x ** 2, 1))

    def planar_flow(self, z, flow_params, num_flows, n_latent_dim, invert_condition=True):
        us, ws, bs = flow_params
        print "us shape:", us.get_shape()
        print "ws shape:", ws.get_shape()
        print "bs shape:", bs.get_shape()
        log_detjs = []
        if num_flows == 0:
            # f_z = z
            sum_logdet_jacobian = logdet_jacobian = 0
            # sum_logdet_jacobian = tf.Constant(0, dtype=tf.float32)
            # logdet_jacobian =  tf.zeros(shape=)
        else:
            for k in range(num_flows):
                u, w, b = us[:, k * n_latent_dim:(k + 1) * n_latent_dim], \
                          ws[:, k * n_latent_dim:(k + 1) * n_latent_dim], bs[:, k]
                print "u shape", u.get_shape()
                print "w shape", w.get_shape()
                print "z shape", z.get_shape()
                print "b shape", b.get_shape()
                if invert_condition:
                    uw = tf.reduce_sum(tf.multiply(w, u), axis=1, keep_dims=True)  # u: (?,2), w: (?,2), b: (?,)
                    # uw = tf.tensordot(u, w, axes = 1)
                    print "uw shape", uw.get_shape()
                    muw = -1 + tf.nn.softplus(uw)  # = -1 + T.log(1 + T.exp(uw))
                    print "muw shape", muw.get_shape()
                    u_hat = u + tf.multiply((muw - uw), w) / tf.norm(w, axis=-1, keep_dims=True)**2
                    print "u_hat shape", u_hat.get_shape()
                else:
                    u_hat = u
                print "u_hat shape", u_hat.get_shape()
                print "tf.cast(z, tf.float32) shape", tf.cast(z, tf.float32).get_shape()
                print "w shape:", w.get_shape()
                zw = tf.reduce_sum(tf.multiply(tf.cast(z, tf.float32), w), axis=1)
                # print "zw shape", zw.get_shape()
                zwb = zw + b
                # print "zwb shape", zwb.get_shape()
                # Equation 10: f(z)= z+ uh(w'z+b)
                # print "u_hat", u_hat.get_shape()
                z = z + u_hat * tf.reshape(self.tanh(zwb), [-1, 1])  # self.z is (?,2)
                psi = tf.reshape((1 - self.tanh(zwb) ** 2), [-1, 1]) * w  # Equation 11. # tanh(x)dx = 1 - tanh(x)**2
                # psi= tf.reduce_sum(tf.matmul(tf.transpose(1-self.tanh(zwb)**2), self.w))
                psi_u = tf.reduce_sum(tf.multiply(u_hat, psi), axis=1, keep_dims=True)
                # psi_u= tf.matmul(tf.transpose(u_hat), tf.transpose(psi))
                # Second term in equation 12. u_transpose*psi_z
                logdet_jacobian = tf.log(tf.abs(1 + psi_u) + 1e-10)  # Equation 12
                # logdet_jacobian = tf.log(tf.clip_by_value(tf.abs(1 + psi_u), 1e-4, 1e7))  # Equation 12
                # print "f_z shape", f_z.get_shape()
                log_detjs.append(logdet_jacobian)
            logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)
            print "logdet_jacobian inside Normalizing Planar flow:", logdet_jacobian.get_shape()
            sum_logdet_jacobian = tf.reduce_sum(logdet_jacobian, axis=1)
            print "sum_logdet_jacobian inside Normalizing Planar flow:", sum_logdet_jacobian.get_shape()
        return z, sum_logdet_jacobian

