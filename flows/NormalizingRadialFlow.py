import tensorflow as tf


class NormalizingRadialFlow(object):
    def __init__(self, z, z_dim=2):
        self.z = z
        self.z_dim = z_dim

    @staticmethod
    def softplus(x):
        return tf.nn.softplus(x)

    @staticmethod
    def get_h(r, alpha):
        h = 1 / (alpha + r)
        return h

    @staticmethod
    def get_derivative_h(r, alpha):
        h_prime = - 1 / ((alpha + r) ** 2)
        return h_prime

    def radial_flow(self, z, flow_params, num_flows, n_latent_dim, invert_condition=True):
        z0s, alphas, betas = flow_params
        print "z0s shape:", z0s.get_shape()
        print "alphas shape:", alphas.get_shape()
        print "betas shape:", betas.get_shape()

        log_detjs = []
        if num_flows == 0:
            # f_z = z
            sum_logdet_jacobian = 0
        else:
            for k in range(num_flows):
                # z0, alpha, beta = z0s[:, k*Z:(k+1)*Z], alphas[:, k*Z:(k+1)*Z], betas[:, k]
                z0, alpha, beta = z0s[:, k * n_latent_dim:(k + 1) * n_latent_dim], alphas[:, k], betas[:, k]
                print "z0 shape", z0.get_shape()
                print "alpha shape", alpha.get_shape()
                print "beta shape", beta.get_shape()
                if invert_condition:
                    # m(x)= log(1 + exp(x)) where x= w'*u. Last equation in A.2 Radial Flows.
                    m_of_beta = self.softplus(beta)
                    print "m_of_beta", m_of_beta.get_shape()
                    print "alpha", alpha.get_shape()
                    beta_hat = -alpha + m_of_beta  # It's a scalar.
                    print "beta_hat", beta_hat.get_shape()
                else:
                    beta_hat = beta
                    print "beta_hat", beta_hat.get_shape()

                # beta_hat = tf.expand_dims(beta_hat,1)
                # Distance of each data point from z0
                dist = (z - z0) ** 2
                dist = tf.reduce_sum(dist, 1)
                r = tf.sqrt(dist)
                # r= tf.sqrt(np.sum(((self.z-self.z0)**2),1))
                # m_of_beta = self.softplus(self.beta) # m(x)= log(1 + exp(x)) where x= w'*u. Last equation in A.2 Radial Flows.
                # beta_hat = -self.alpha + m_of_beta # It's a scalar.

                h_alpha_r = self.get_h(r, alpha)  # Argument of h(.) in equation 14. (1000000,)
                print "beta_hat", beta_hat.get_shape()
                beta_h_alpha_r = beta_hat * h_alpha_r
                print "beta_h_alpha_r", beta_h_alpha_r.get_shape()
                # fz = self.z + beta_hat * tf.mul(tf.transpose(tf.expand_dims(h_alpha_r, 1)),
                #                                            (self.z-self.z0))
                print "h_alpha_r shape", h_alpha_r.get_shape()
                # print "h_alpha_r shape", tf.expand_dims(h_alpha_r,1).get_shape()
                print "z shape", z.get_shape()
                # z = z + beta_hat * tf.multiply((z-z0), h_alpha_r)
                print "Shape 2nd term", tf.multiply(tf.multiply((z - z0), h_alpha_r), beta_hat).get_shape()
                # z = z + tf.multiply(tf.multiply((z-z0), h_alpha_r), beta_hat)
                z = z + tf.multiply(tf.multiply((z - z0), tf.expand_dims(h_alpha_r, 1)), tf.expand_dims(beta_hat, 1))
                # print "z shape", z.get_shape()
                # Calculation of log det jacobian
                print "r shape", r.get_shape()
                print "alpha shape", alpha.get_shape()

                h_derivative_alpha_r = self.get_derivative_h(r, alpha)
                beta_h_derivative_alpha_r = beta_hat * h_derivative_alpha_r
                print "h_derivative_alpha_r shape", h_derivative_alpha_r.get_shape()
                print "beta_h_derivative_alpha_r shape", beta_h_derivative_alpha_r.get_shape()

                logdet_jacobian = tf.multiply(((1 + beta_h_alpha_r) ** (n_latent_dim - 1)),
                                              (1 + beta_h_derivative_alpha_r * r + beta_h_alpha_r))
                print "logdet_jacobian shape", logdet_jacobian.get_shape()
                log_detjs.append(tf.expand_dims(logdet_jacobian, 1))
            logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)
            print "logdet_jacobian inside Normalizing Planar flow:", logdet_jacobian.get_shape()
            sum_logdet_jacobian = tf.reduce_mean(logdet_jacobian, axis=0)
            print "sum_logdet_jacobian inside Normalizing Planar flow:", sum_logdet_jacobian.get_shape()
            print "z shape", z.get_shape()
        return z, sum_logdet_jacobian


