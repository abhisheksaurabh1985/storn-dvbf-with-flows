import numpy as np
import tensorflow as tf
import scipy
import scipy.linalg   # SciPy Linear Algebra Library


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
    def get_indices_of_diagonal_elements(n):
        """
        Jacoabian matrix in case of Convolution Planar flow is n*n matrix.
        :param n: Number of columns
        :return: Indices of the diagonal elements. Diagonal implies main, first and second upper and lower diagonals.
        """
        indices_diagonal, indices_first_upper_diagonal, \
        indices_second_upper_diagonal, indices_first_lower_diagonal, indices_second_lower_diagonal = [], [], [], [], []

        # n = 200

        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", n

        for i in range(n):
            indices_diagonal.append(i * (n + 1))

            if i == (n-1):
                pass
            else:
                indices_first_upper_diagonal.append(i * (n + 1) + 1)

            if i == (n-1) or i == (n-2):
                pass
            else:
                indices_second_upper_diagonal.append(i * (n + 1) + 2)

            if i == 0:
                pass
            else:
                indices_first_lower_diagonal.append(i * (n + 1) - 1)
            if i == 0 or i == 1:
                pass
            else:
                indices_second_lower_diagonal.append(i * (n + 1) - 2)

        return indices_diagonal, indices_first_upper_diagonal, indices_second_upper_diagonal, \
               indices_first_lower_diagonal, indices_second_lower_diagonal

    @staticmethod
    def get_LU_decomposition(A):
        P, L, U = scipy.linalg.lu(A)
        return P, L, U

    @staticmethod
    def conv1d(z, W, strides=1):
        """
        z shape: (bs, ts, ?), w shape: (3, 2, 1). input_channel =2 for the filter corresponds to the two dimensions of
        the input data. output_channel = 1. Expected output with padding and stride = 1 is (bs, ts, 1).

        Padding options: 'SAME' will output the same input length. 'VALID' is other option which does not add zero
        padding.

        :param z: Data in the latent space with shape (bs, ts, ?).
        :param W: Filter with shape: (3, 2, 1). input_channel =2 for the filter corresponds to the two dimensions of
        the input data. output_channel = 1.
        :param strides: Integer.
        :return: Convolved 1D signal with shape (bs, ts, 1)
        """
        # Conv1D wrapper
        temp = tf.nn.conv1d(z, W, stride=strides, padding="SAME")
        return temp

    @staticmethod
    def make_jacobian(y, x, num_obs):
        """
        For reshape=="3D" case.
        :param y:
        :param x:
        :param num_obs:
        :return:
        """

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
        :return: # (?, ?, 200). The second ? refers to 200.
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
        ] # This is a list of tensors that is passed to both cond and body.

        def body(j, result):
            """
            body is a callable returning a (possibly nested) tuple, namedtuple or list of tensors of the same arity
            (length and structure) and types as loop_vars.
            """
            if tf.gradients(y[:, j], x)[0] is not None:
                return j + 1, result.write(j, tf.reshape(tf.gradients(y[:, j], x)[0], shape=[-1, num_obs]))
            else:
                return j + 1, result.write(j, tf.reshape(tf.zeros_like(x, dtype=tf.float32), shape=[-1, num_obs]))

        def cond(j, result):
            """
            cond is a callable returning a boolean scalar tensor. cond and body both take as many arguments as there are
            loop_vars. Hence, we have to pass 'result' here despite the fact that result will not be used here.
            """
            return j < num_obs

        # tf.while_loo: Repeat body while the condition cond is true.
        _, jacobian = tf.while_loop(cond, body, loop_vars)  # Shape of jacobian: (?, 20, 200)
        print "Jacobian shape inside make_jacobian function", jacobian.stack().get_shape()
        return tf.transpose(jacobian.stack(), [1, 0, 2])

    def convolution_planar_flow_with_jacobian_manually_calculated(self, z, flow_params,
                                                                  num_flows, n_latent_dim, filter_width=3,
                                                                  invert_condition=False, reshape="2D"):
        """
        USE THIS ONE.

        z shape: (bs,ts,?); us, bs shape: (ts, bs, ?); ws shape: (1, num_flows * filter_width * n_latent_dim * 1)

        reshape: Switch added just for testing purposes. "2D" should be used. "3D" added for testing purposes.
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

            # Get indices of the diagonal matrix in the jacobian matrix which will be calculated later.
            number_of_columns_in_jacobian_matrix = self.time_steps * self.z_dim
            n_col = tf.constant(number_of_columns_in_jacobian_matrix, dtype=tf.int32, name="ncol_jacobian_matrix")
            idx_diag, idx_first_upper_diag, idx_second_upper_diag, \
            idx_first_lower_diag, idx_second_lower_diag = \
                tf.py_func(self.get_indices_of_diagonal_elements, [n_col], [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])

            print "############### Length of lists ######################"
            print idx_diag.get_shape() # len(idx_first_upper_diag), len(idx_second_upper_diag), len(idx_first_lower_diag), \
                  # len(idx_second_lower_diag)

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
                # wz = self.conv1d(z, self.tf_rot180(w))
                print "wz shape:", wz.get_shape()
                print "b with dims expanded:", tf.expand_dims(b, 2)

                # Step 2:
                # Extend dimension of b. Original b shape: (bs, ts). After expanding dimensions it is (bs, ts, 1).
                wzb = wz + tf.expand_dims(b, 2)  # wzb shape: (bs, ts, 1)
                print "wzb shape:", wzb.get_shape()

                # Step 3:
                print "tanh(wzb) shape:", tf.nn.tanh(wzb).get_shape()
                f_z = z + u * tf.nn.tanh(wzb, name="tanh_wzb")  # transformed z: (bs,ts,?)
                # f_z = u * tf.nn.tanh(wzb, name="tanh_wzb")
                print "transformed z shape:", z.get_shape()
                print "dtanh wzb shape:", self.dtanh(wzb).get_shape()  # (bs, ts, 1)

                # psi = self.conv1d(self.dtanh(wzb), self.tf_rot180(tf.transpose(w, perm=[0, 2, 1])))
                # print "psi shape:", psi.get_shape()
                # print "u shape:", u.get_shape()
                # psi_ut = tf.matmul(u, tf.transpose(psi, perm=[0, 2, 1], name="psi_transpose"))
                # print "psi_ut shape:", psi_ut.get_shape()

                # Step 4: Jacobian matrix
                if reshape == "2D":
                    # USE THIS ONE
                    print "f_z shape", f_z.get_shape()
                    print "z shape", z.get_shape()
                    reshaped_f_z = tf.reshape(f_z, shape=[self.batch_size, self.time_steps * self.z_dim])
                    print "reshaped f_z shape", reshaped_f_z.get_shape()
                    # Note that in the following line the dependent variable z has not been reshaped.
                    jacobian_z = self.make_jacobian_2D(reshaped_f_z, z, self.time_steps * self.z_dim)  # (?, ?, 200). The second ? refers to 200.
                elif reshape == "3D":
                    # Without reshaping
                    jacobian_z = self.make_jacobian(f_z, z, self.z_dim)  # (?, ?, 200)

                print "=========================================="
                print "jacobian_z shape", jacobian_z.get_shape()
                print "=========================================="

                # Determinant using tf.matrix_determinant
                # Determinant. Without expanding dims, the shape is (b,t). Post expansion, shape is (b,t,1).
                jacobian_z = tf.Print(jacobian_z, [tf.constant("jacobian_z"), tf.reduce_max(jacobian_z),
                                                   tf.reduce_min(jacobian_z), tf.reduce_mean(jacobian_z)])

                main_diagonal = tf.matrix_diag_part(jacobian_z, "main_diagonal")
                print "main diagonal shape:", main_diagonal.get_shape()

                # First upper diagonal. Remove the first column and last row. Follow a similar process for second upper,
                # first and second lower diagonal as well.
                first_upper_diagonal = tf.matrix_diag_part(tf.slice(jacobian_z,
                                                                    begin=[0, 0, 1],
                                                                    size=[-1, (self.time_steps*self.z_dim)-1,
                                                                              (self.time_steps*self.z_dim)],
                                                                    name="slice_for_first_upper_diagonal"),
                                                           name="first_upper_diagonal")
                print "first upper diagonal shape:", first_upper_diagonal.get_shape()

                # Second upper diagonal
                second_upper_diagonal = tf.matrix_diag_part(tf.slice(jacobian_z,
                                                                     begin=[0, 0, 2],
                                                                     size=[-1, (self.time_steps*self.z_dim)-2,
                                                                               (self.time_steps*self.z_dim)],
                                                                     name="slice_for_second_upper_diagonal"),
                                                            name="first_upper_diagonal")
                print "second upper diagonal shape:", second_upper_diagonal.get_shape()

                # First lower diagonal
                first_lower_diagonal = tf.matrix_diag_part(tf.slice(jacobian_z,
                                                                    begin=[0, 1, 0],
                                                                    size=[-1, (self.time_steps*self.z_dim),
                                                                              (self.time_steps*self.z_dim)-1],
                                                                    name="slice_for_first_lower_diagonal"),
                                                           name="first_upper_diagonal")
                print "first lower diagonal shape:", first_lower_diagonal.get_shape()

                # Second lower diagonal
                second_lower_diagonal = tf.matrix_diag_part(tf.slice(jacobian_z,
                                                                     begin=[0, 2, 0],
                                                                     size=[-1, (self.time_steps*self.z_dim),
                                                                              (self.time_steps*self.z_dim)-2],
                                                                     name="slice_for_second_lower_diagonal"),
                                                            name="second_upper_diagonal")
                print "second lower diagonal shape:", second_lower_diagonal.get_shape()

                # determinant_jacobian = tf.reduce_prod(main_diagonal, axis=1, name="determinant_jacobian")
                determinant_jacobian = tf.matrix_determinant(jacobian_z, name="jacobian_matrix_determinant")
                print "determinant_jacobian shape:", determinant_jacobian.get_shape()  # (?,)
                logdet_jacobian = tf.log(tf.abs(determinant_jacobian, name="absolute_value_logdet") + 1e-5,
                                         name="logdet_jacobian")
                print "logdet_jacobian shape:", logdet_jacobian.get_shape()  # (?,)

                # _logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10))
                log_detjs.append(tf.expand_dims(logdet_jacobian, axis=1))
                # log_detjs.append(logdet_jacobian)
                print "shape of first element in list:", tf.shape(log_detjs[0])
                print "shape of first element in list:", log_detjs[0].get_shape()  # (?, 1)
            # Concatenated vertically. Below, logdet_jacobian shape: (bs, numFlows*ts, 2).
            logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)
            print "#######@@@@@@@@@@!!!!!!!!!!!1##########@@@@@@@@@@!!!!!!!!!"
            print "logdet_jacobian inside Convolution Planar flow:", logdet_jacobian.get_shape()  # (?, 4)
            sum_logdet_jacobian = tf.reduce_sum(logdet_jacobian, axis=1)  # shape: ()
            print "#######@@@@@@@@@@ !!!!!!!!!!! ##########@@@@@@@@@@!!!!!!!!!"
            print "sum_logdet_jacobian inside Convolution Planar flow:", sum_logdet_jacobian.get_shape()
        return tf.transpose(z, perm=[1, 0, 2]), sum_logdet_jacobian, jacobian_z


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


        # def convolution_planar_flow(self, z, flow_params, num_flows, n_latent_dim, filter_width=3, invert_condition=False):
    #     """
    #     THIS IS NOT BEING USED. USE THE FUNCTION WHICH FOLLOWS.
    #
    #     z shape: (bs,ts,?); us, bs shape: (ts, bs, ?); ws shape: (1, num_flows * filter_width * n_latent_dim * 1)
    #     """
    #     us, ws, bs = flow_params
    #     print "us shape:", us.get_shape()
    #     print "ws shape:", ws.get_shape()
    #     print "bs shape:", bs.get_shape()
    #     print "z shape:", z.get_shape()
    #     log_detjs = []
    #     if num_flows == 0:
    #         sum_logdet_jacobian = tf.Variable(0.0, dtype=tf.float32)
    #     else:
    #         for k in range(num_flows):
    #             u, w, b = us[:, :, k * n_latent_dim:(k + 1) * n_latent_dim], \
    #                       ws[:, k * filter_width * n_latent_dim * 1:(k + 1) * filter_width * n_latent_dim * 1], \
    #                       bs[:, :, k]
    #
    #             # Reshape u and b s.t. batch size is the first dimension. When applicable, n_time_step is the second
    #             # dimension and n_latent_dim the third.
    #             print "$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#$#"
    #             print "b before transpose:", b.get_shape()
    #             u = tf.transpose(u, perm=[1, 0, 2])  # u shape: (20, 100, ?)
    #             b = tf.transpose(b, perm=[1, 0])  # (20, 100)
    #             print "u shape:", u.get_shape()
    #             print "w shape:", w.get_shape()
    #             print "b shape:", b.get_shape()
    #             w = tf.reshape(w, shape=[filter_width, n_latent_dim, 1])  # w shape: (1, 6); reshaped w shape: (3, 2, 1)
    #             print "reshaped w shape:", w.get_shape()
    #
    #             # Step 1: Flip the filter by 180 degrees and then convolve. If filter isn't rotated, then tf.nn.conv2d,
    #             # which is actually called by tf.nn.conv1d, performs a correlation rather than convolution.
    #             # z shape: (bs, ts, ?), w shape: (3, 2, 1)
    #             wz = self.conv1d(z, w)  # wz shape: (bs, ts, 1)
    #             print "wz shape:", wz.get_shape()
    #             print "b with dims expanded:", tf.expand_dims(b, 2)
    #
    #             # Step 2:
    #             # Extend dimension of b. Original b shape: (bs, ts). After expanding dimensions it is (bs, ts, 1).
    #             wzb = wz + tf.expand_dims(b, 2)  # wzb shape: (bs, ts, 1)
    #             print "wzb shape:", wzb.get_shape()
    #
    #             # Step 3:
    #             print "tanh(wzb) shape:", tf.nn.tanh(wzb).get_shape()
    #             z = z + u * tf.nn.tanh(wzb, name="tanh_wzb")  # transformed z: (bs,ts,?)
    #             print "transformed z shape:", z.get_shape()
    #             print "dtanh wzb shape:", self.dtanh(wzb).get_shape()  # (bs, ts, 1)
    #
    #             # Step 4: First transpose the filter to match the dimensions of psi. Then rotate it for convolution.
    #             # psi shape: (bs, ts, 2)
    #             psi = self.conv1d(self.dtanh(wzb), self.tf_rot180(tf.transpose(w, perm=[0, 2, 1])))
    #             print "psi shape:", psi.get_shape()
    #
    #             # Step 5: This is correct. Jacobian of psi_ut is to be calculated.
    #             # u shape: (bs, ts, ?), psi shape: (bs, ts, 2), psi_ut:(bs, ts, 2)
    #             # psi_ut = tf.multiply(psi, u, name="psi_ut")
    #             # psi_ut = tf.transpose(tf.matmul(tf.transpose(u, perm=[0, 2, 1]), psi, name="psi_u"), perm=[0, 2, 1],
    #             #                       name="psi_ut")  # Reshaped from (bs, ?, 1) to (bs, 1, ?).
    #             psi_ut = tf.matmul(u, tf.transpose(psi, perm=[0, 2, 1], name="psi_transpose"))
    #                                    # Reshaped from (bs, ?, 1) to (bs, 1, ?).
    #             print "psi_ut shape:", psi_ut.get_shape()
    #
    #             # Step 6:
    #             logdet_jacobian = tf.log(tf.abs(1 + psi_ut) + 1e-10)  # logdet_jacobian shape: (bs, ts, 2)
    #             # singular_value = tf.stop_gradient(tf.svd(psi_ut, compute_uv=False))
    #             # tf.stop_gradient(singular_value, name="stop_gradient_svd")
    #             # singular_value = np.linalg.svd(psi_ut, compute_uv=0)
    #             # logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10), axis=1)
    #             print "logdet_jacobian shape:", logdet_jacobian.get_shape()
    #             # _logdet_jacobian = tf.reduce_sum(tf.log(singular_value + 1e-10))
    #             # log_detjs.append(tf.expand_dims(logdet_jacobian, axis=1))
    #             log_detjs.append(logdet_jacobian)
    #             print "shape of first element in list:", log_detjs[0].get_shape()
    #         # Concatenated vertically. Below, logdet_jacobian shape: (bs, numFlows*ts, 2).
    #         logdet_jacobian = tf.concat(log_detjs[0:num_flows + 1], axis=1)
    #         print "logdet_jacobian inside Normalizing Planar flow:", logdet_jacobian.get_shape()
    #         sum_logdet_jacobian = tf.reduce_sum(logdet_jacobian)  # shape: ()
    #         print "sum_logdet_jacobian inside Normalizing Planar flow:", sum_logdet_jacobian.get_shape()
    #     return tf.transpose(z, perm=[1, 0, 2]), sum_logdet_jacobian
