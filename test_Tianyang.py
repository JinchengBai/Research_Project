import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.getcwd() + "/output/"
EXP = "072718-0"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 1000
X_dim = 1  # dimension of the target distribution, 3 for e.g.
z_dim = 1
h_dim_g = 50
h_dim_d = 50
N, n_D, n_G = 2000, 1, 1  # num of iterations


mu1 = 1
Sigma1 = 1
Sigma1_inv = 1/Sigma1
# mu1 = np.ones(X_dim)
# Sigma1 = np.identity(X_dim)
# Sigma1_inv = np.linalg.inv(Sigma1)

################################################################################################
################################################################################################


def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "info.txt", 'w')
info.write("Description: " + '\n' +
           "KSD training"
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['mu1 = {}'.format(mu1), output_matrix('sigma1 = {}', Sigma1)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d), 'n_D = {}'.format(n_D),
                        'n_G = {}'.format(n_G)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Training iter: \n\t" +
           "\n\t".join(['n_D = {}'.format(n_D), 'n_G = {}'.format(n_G)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n'
           "Additional Information: \n" +
           "" + "\n")
info.close()


################################################################################################
################################################################################################
# convert parameters to tf tensor
mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[1])
Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[1, 1])

X = tf.placeholder(tf.float32, shape=[None, X_dim])

initializer = tf.contrib.layers.xavier_initializer()

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [X_dim], initializer=initializer)

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, X_dim], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [X_dim], initializer=initializer)

theta_G = [G_W1, G_b1]


# def log_densities(xs):
#     log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1, Sigma1_inv),
#                                         tf.transpose(xs - mu1))) / 2
#     return log_den1


def S_q(xs):
    return tf.matmul(mu1_tf - xs, Sigma1_inv_tf)
    # return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n):
    # np.random.seed(1)
    return np.random.normal(0, 1, size=[m, n])


def generator(z):
    G_h1 = (tf.matmul(z, G_W1) + G_b1)
    return G_h1


def svgd_kernel(x, dim=X_dim, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = tf.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), 1)
    dxkxy = tf.add(-tf.matmul(kxy, x), tf.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    dxykxy_tr = tf.multiply((dim * (h**2) - pdist), kxy)  # tr( dk(x, y)/dxdy )

    return kxy, dxkxy, dxykxy_tr


def ksd_emp(x, n=mb_size, dim=X_dim, h=1.):  # credit goes to Hanxi!!! ;P
    sq = S_q(x)
    kxy, dxkxy, dxykxy_tr = svgd_kernel(x, dim, h)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))

    phi = (tf.matmul(kxy, sq) + dxkxy) / n

    return ksd, phi


def phi_func(y, x, h=1.):
    """
    This function evaluates the optimal phi from KSD at any point
    :param y: evaluate phi at y, dimension m * d
    :param x: data set used to calculate empirical expectation, dimension n*d
    :param h: the parameter in kernel function
    :return: the value of dimension m * d
    """
    m = tf.shape(y)[0]
    n = tf.shape(x)[0]
    XY = tf.matmul(y, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[n, 1])
    X2 = tf.tile(X2_, [1, m])
    Y2_ = tf.reshape(tf.reduce_sum(tf.square(y), axis=1), shape=[m, 1])
    Y2 = tf.tile(Y2_, [1, n])
    pdist = tf.subtract(tf.add(Y2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = tf.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), 1)
    dxkxy = tf.add(-tf.matmul(kxy, x), tf.multiply(y, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    phi = (tf.matmul(kxy, S_q(x)) + dxkxy) / tf.cast(n, dtype=tf.float32)

    return phi


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)

ksd, D_fake = ksd_emp(G_sample)

# phi_y = phi_func(G_sample, G_sample)


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# x, f, p = sess.run([G_sample, D_fake, phi_y], feed_dict={z: sample_z(mb_size, z_dim)})
# print(f)
# print(p)


range_penalty_g = 10*(generator(tf.constant(1, shape=[1, 1], dtype=tf.float32)) -
                      generator(tf.constant(-1, shape=[1, 1], dtype=tf.float32)))
# range_penalty_g = tf.Print(range_penalty_g, [range_penalty_g], message="range_penalty_g"+"-values:")


loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
loss2 = diag_gradient(D_fake, G_sample)

Loss = tf.reduce_mean(loss1 + loss2)


G_solver = (tf.train.AdamOptimizer(learning_rate=1e-2).minimize(Loss, var_list=theta_G))

# G_solver_ksd = (tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(ksd, var_list=theta_G))
G_solver_ksd = (tf.train.AdamOptimizer(learning_rate=1e-2).minimize(ksd, var_list=theta_G))


#######################################################################################################################
#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

ksd_loss = np.zeros(N)
G_loss = np.zeros(N)
ksd_curr = G_Loss_curr = None

for it in range(N):
    for _ in range(n_G):
        _, G_Loss_curr, ksd_curr = sess.run([G_solver_ksd, Loss, ksd],
                                            feed_dict={z: sample_z(mb_size, z_dim)})
    G_loss[it] = G_Loss_curr
    ksd_loss[it] = ksd_curr

    if np.isnan(G_Loss_curr):
        print("G_loss:", it)
        break

    if it % 10 == 0:
        noise = sample_z(100, 1)
        x_range = np.reshape(np.linspace(-5, 5, 500, dtype=np.float32), newshape=[500, 1])
        z_range = np.reshape(np.linspace(-5, 5, 500, dtype=np.float32), newshape=[500, 1])
        samples = sess.run(generator(noise.astype(np.float32)))

        gen_func = sess.run(generator(z_range))
        disc_func = sess.run(phi_func(z_range, samples))

        sample_mean = np.mean(samples)
        sample_sd = np.std(samples)
        print(it, ":", sample_mean, sample_sd)
        print("ksd_loss:", ksd_curr)
        print("G_loss:", G_Loss_curr)
        print("w:", G_W1.eval(session=sess), "b:", G_b1.eval(session=sess))
        # plt.scatter(samples[:, 0], samples[:, 1], color='b')
        # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
        plt.plot()
        # plt.subplot(212)
        # plt.plot(x_range, disc_func)
        plt.subplot(212)
        plt.ylim(-3, 3)
        plt.plot(x_range, disc_func)
        plt.subplot(211)
        plt.ylim(-3, 5)
        plt.plot(range(100), samples[:, 0], 'ro', color='b', ms=1)
        plt.axhline(mu1, color='r')
        plt.title(
            "iter {0:04d}, {{G: {1:.4f}, mu: {2:.4f}, sd: {3:.4f}}}".format(it, G_Loss_curr, sample_mean, sample_sd))
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()

sess.close()


np.savetxt(EXP_DIR + "loss_ksd.csv", ksd_loss, delimiter=",")
plt.plot(ksd_loss)
plt.ylim(ymin=0)
plt.axvline(np.argmin(ksd_loss), ymax=np.min(ksd_loss), color="r")
plt.title("KSD (min at iter {})".format(np.argmin(ksd_loss)))
plt.savefig(EXP_DIR + "ksd.png", format="png")
plt.close()