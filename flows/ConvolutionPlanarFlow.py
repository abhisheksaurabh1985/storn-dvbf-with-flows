import numpy as np
import tensorflow as tf


class ConvolutionPlanarFlow(object):

    def __init__(self, z, z_dim=2, time_steps=100, batch_size=20):
        self.z = z  # shape (nSamples, nDim)
        self.z_dim = z_dim
        self.time_steps = time_steps
        self.batch_size = batch_size

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

    # @staticmethod
    # def make_jacobian(y, x, num_obs=5):
    #     y_flat = tf.reshape(y, (-1,))
    #     #y_flat = tf.reshape(y, shape=[tf.shape(y)[0], -1])
    #     #print "y_flat shape", y_flat.get_shape()
    #     print 1
    #     jacobian_flat = tf.stack(
    #         [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
    #     print 1
    #     #print "jacobian_flat shape", jacobian_flat.get_shape()
    #     return tf.reshape(jacobian_flat, tf.concat([tf.shape(y), tf.shape(x)], 0))

    @staticmethod
    def make_jacobian(y, x, num_obs):
        n = tf.shape(y)[-1]

        print "@@@@@@@@@@@@@@@@@@@@@@@ make_jacobin @@@@@@@@@@@@@@@@@@@@@@@@@"
        print n
        print "y shape:", y.get_shape()
        print "x shape:", x.get_shape()

        # print len(tf.gradients(y, x))
        # print tf.gradients(y, x)
        # print [tf.gradients(y[:, :, j], x)[0] for j in range(num_obs)]
        #j = tf.constant(0)
        #l = tf.Variable([])

        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=num_obs),
        ]

        def body(j, result):
            # print(result)
            # j = tf.Print(j, [j, result, y, y[:, j]])
            if tf.gradients(y[:, :, j], x)[0] is not None:
                return j + 1, result.write(j, tf.gradients(y[:, :, j], x)[0])
            else:
                # return tf.Variable(0)
                return j + 1, result.write(j, tf.zeros_like(x, dtype=tf.float32))
            # return j + 1, result.write(j, tf.gradients(y[:, j], x)[0])

        def cond(j, result):
            return j < num_obs

        # _, jacobian = tf.while_loop(cond, body, loop_vars, shape_invariants=[l.get_shape(),
        #                                                                      tf.TensorShape([None])])
        _, jacobian = tf.while_loop(cond, body, loop_vars)  #[j, l]) #, shape_invariants=[l.get_shape(),
                                                         #                 tf.TensorShape(None)])
        print "Jacobian shape inside make_jacobian function", jacobian.stack().get_shape()
        return tf.transpose(jacobian.stack(), [1, 2, 0, 3])


    @staticmethod
    def make_jacobian_2D(y, x, num_obs):
        """
        :param y: shape is (b, t*d)
        :param x: shape is (b, t*d)
        :param num_obs: self.time_steps * self.z_dim
        :return:
        """
        n = tf.shape(y)[-1]
        print "@@@@@@@@@@@@@@@@@@@@@@@ make_jacobin @@@@@@@@@@@@@@@@@@@@@@@@@"
        # print n
        # print len(tf.gradients(y, x))  # length is 1.
        # print tf.gradients(y, x)  # [None]
        # print [tf.gradients(y[:, j], x)[0] for j in range(num_obs)]  # List with all elements as None.

        loop_vars = [
            tf.constant(0, tf.int32),
            tf.TensorArray(tf.float32, size=num_obs),
        ]

        # There is some issue with tf.gradients. It returns a list of sum(dy/dx) for each x in xs.
        def body(j, result):
            if tf.gradients(y[:, j], x)[0] is not None:
                return j + 1, result.write(j, tf.reshape(tf.gradients(y[:, j], x)[0], shape=[-1, num_obs]))
            else:
                return j + 1, result.write(j, tf.reshape(tf.zeros_like(x, dtype=tf.float32), shape=[-1, num_obs]))

        def cond(j, result):
            return j < num_obs

        _, jacobian = tf.while_loop(cond, body, loop_vars)  # Shape of jacobian: (?, 20, 200)
        print "Jacobian shape inside make_jacobian function", jacobian.stack().get_shape()
        return tf.transpose(jacobian.stack(), [1, 0, 2])

    def convolution_planar_flow(self, z, flow_params, num_flows, n_latent_dim, filter_width=3, invert_condition=False):
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

                # Step 3:
                print "tanh(wzb) shape:", tf.nn.tanh(wzb).get_shape()
                z = z + u * tf.nn.tanh(wzb, name="tanh_wzb")  # transformed z: (bs,ts,?)
                print "transformed z shape:", z.get_shape()
                print "dtanh wzb shape:", self.dtanh(wzb).get_shape()  # (bs, ts, 1)

                # Step 4: First transpose the filter to match the dimensions of psi. Then rotate it for convolution.
                # psi shape: (bs, ts, 2)
                psi = self.conv1d(self.dtanh(wzb), self.tf_rot180(tf.transpose(w, perm=[0, 2, 1])))
                print "psi shape:", psi.get_shape()

                # Step 5: This is correct. Jacobian of psi_ut is to be calculated.
                # u shape: (bs, ts, ?), psi shape: (bs, ts, 2), psi_ut:(bs, ts, 2)
                # psi_ut = tf.multiply(psi, u, name="psi_ut")
                # psi_ut = tf.transpose(tf.matmul(tf.transpose(u, perm=[0, 2, 1]), psi, name="psi_u"), perm=[0, 2, 1],
                #                       name="psi_ut")  # Reshaped from (bs, ?, 1) to (bs, 1, ?).
                psi_ut = tf.matmul(u, tf.transpose(psi, perm=[0, 2, 1], name="psi_transpose"))
                                       # Reshaped from (bs, ?, 1) to (bs, 1, ?).
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

    def convolution_planar_flow_with_jacobian_manually_calculated(self, z, flow_params,
                                                                  num_flows, n_latent_dim, filter_width=3,
                                                                  invert_condition=False, reshape="3D"):
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

                # Step 3:
                print "tanh(wzb) shape:", tf.nn.tanh(wzb).get_shape()
                f_z = z + u * tf.nn.tanh(wzb, name="tanh_wzb")  # transformed z: (bs,ts,?)
                print "transformed z shape:", z.get_shape()
                print "dtanh wzb shape:", self.dtanh(wzb).get_shape()  # (bs, ts, 1)

                if reshape == "2D":
                    # Step 4: This is the calculation of Jacobian. If this works, then calculation of psi and psi_ut may
                    # not be needed.
                    print "f_z shape", f_z.get_shape()
                    print "z shape", z.get_shape()
                    # jacobian_z = self.jacobian(tf.reshape(f_z, shape=[z.get_shape().as_list()[0], -1]), z)
                    # reshaped_f_z = tf.reshape(f_z, shape=[tf.shape(z)[0], self.time_steps * self.z_dim])
                    reshaped_f_z = tf.reshape(f_z, shape=[self.batch_size, self.time_steps * self.z_dim])
                    # reshaped_f_z = tf.reshape(f_z, shape=[z.get_shape().as_list()[0], 200])
                    print "reshaped f_z shape", reshaped_f_z.get_shape()

                    # reshaped_z = tf.reshape(z, shape=[tf.shape(z)[0], self.time_steps * self.z_dim])
                    # reshaped_z = tf.reshape(z, shape=[self.batch_size, self.time_steps * self.z_dim])
                    # print "reshaped z shape", reshaped_z.get_shape()
                    jacobian_z = self.make_jacobian_2D(reshaped_f_z,
                                                       z,
                                                       self.time_steps * self.z_dim)
                elif reshape == "3D":
                    # Without reshaping
                    jacobian_z = self.make_jacobian(f_z,
                                                    z,
                                                    self.z_dim)  # (20, 100, ?, ?)
                print "jacobian_z shape", jacobian_z.get_shape()  # (?, ?, 200)
                # # Determinant. Without expanding dims, the shape is (b,t). Post expansion, shape is (b,t,1).
                # # determinant_jacobian = tf.matrix_determinant(jacobian_z, name="determinant_jacobian")
                determinant_jacobian = tf.expand_dims(tf.matrix_determinant(jacobian_z,
                                                                            name="determinant_jacobian"), -1)
                # print "determinant_jacobian shape", determinant_jacobian.get_shape()
                # Step 4: First transpose the filter to match the dimensions of psi. Then rotate it for convolution.
                # psi shape: (bs, ts, 2)
                # psi = self.conv1d(self.dtanh(wzb), self.tf_rot180(tf.transpose(w, perm=[0, 2, 1])))
                # print "psi shape:", psi.get_shape()
                #
                # # Step 5: This is correct. Jacobian of psi_ut is to be calculated.
                # # u shape: (bs, ts, ?), psi shape: (bs, ts, 2), psi_ut:(bs, ts, 2)
                # # psi_ut = tf.multiply(psi, u, name="psi_ut")
                # # psi_ut = tf.transpose(tf.matmul(tf.transpose(u, perm=[0, 2, 1]), psi, name="psi_u"), perm=[0, 2, 1],
                # #                       name="psi_ut")  # Reshaped from (bs, ?, 1) to (bs, 1, ?).
                # psi_ut = tf.matmul(u, tf.transpose(psi, perm=[0, 2, 1], name="psi_transpose"))
                #                        # Reshaped from (bs, ?, 1) to (bs, 1, ?).
                # print "psi_ut shape:", psi_ut.get_shape()
                #
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # Step 6:
                # logdet_jacobian = tf.log(tf.abs(1 + psi_ut) + 1e-10)  # logdet_jacobian shape: (bs, ts, 2)
                logdet_jacobian = tf.log(tf.abs(1 + determinant_jacobian) + 1e-10)  # logdet_jacobian shape: (bs, ts, 2)
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # singular_value = tf.stop_gradient(tf.svd(determinant_jacobian, compute_uv=False))
                # tf.stop_gradient(singular_value, name="stop_gradient_svd")
                # singular_value = np.linalg.svd(determinant_jacobian, compute_uv=0)
                # logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10), axis=1)
                print "logdet_jacobian shape:", logdet_jacobian.get_shape()
                # _logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10))
                # log_detjs.append(tf.expand_dims(logdet_jacobian, axis=1))
                log_detjs.append(logdet_jacobian)
                print "shape of first element in list:", log_detjs[0].get_shape()
            # Concatenated vertically. Below, logdet_jacobian shape: (bs, numFlows*ts, 2).
            logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)
            print "#######@@@@@@@@@@!!!!!!!!!!!1##########@@@@@@@@@@!!!!!!!!!"
            print "logdet_jacobian inside Convolution Planar flow:", logdet_jacobian.get_shape()
            sum_logdet_jacobian = tf.reduce_sum(logdet_jacobian)  # shape: ()
            print "#######@@@@@@@@@@!!!!!!!!!!!1##########@@@@@@@@@@!!!!!!!!!"
            print "sum_logdet_jacobian inside Convolution Planar flow:", sum_logdet_jacobian.get_shape()
        return tf.transpose(z, perm=[1, 0, 2]), sum_logdet_jacobian, jacobian_z

