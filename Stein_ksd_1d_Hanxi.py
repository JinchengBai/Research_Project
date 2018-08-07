"""

Stein_GAN: Stein_ksd_1d_Hanxi.py

Created on 7/25/18 8:20 PM

@author: Hanxi Sun

"""

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.getcwd() + "/output/"
EXP = "072618-1"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 3
X_dim = 2  # dimension of the target distribution, 3 for e.g.
z_dim = X_dim
h_dim_g = 50
h_dim_d = 50
N, n_D, n_G = 5000, 1, 1  # num of iterations


# mu1 = 1
# Sigma1 = 1
# Sigma1_inv = 1/Sigma1
mu1 = np.ones(X_dim)
Sigma1 = np.identity(X_dim)
Sigma1_inv = np.linalg.inv(Sigma1)

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
mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[X_dim])
Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[X_dim, X_dim])

X = tf.placeholder(tf.float32, shape=[None, X_dim])

initializer = tf.contrib.layers.xavier_initializer()

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [X_dim], initializer=initializer)

theta_D = [D_W1, D_W2, D_b1, D_b2]


# z = tf.placeholder(tf.float32, shape=[None, z_dim])
z = tf.placeholder(tf.float32, shape=[mb_size, z_dim])

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
    return G_h1, G_W1


def svgd_kernel(x, dim=X_dim, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    pdist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = tf.exp(- pdist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), - 1)
    dxkxy = tf.add(-tf.matmul(kxy, x), tf.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    dxykxy_tr = tf.multiply((dim * (h**2) - pdist), kxy)  # tr( dk(x, y)/dxdy )

    return kxy, dxkxy, dxykxy_tr  # , tf.gradients(kxy, x)


def ksd_emp(x, n=mb_size, dim=X_dim, h=1.):  # credit goes to Hanxi!!! ;P
    sq = S_q(x)
    kxy, dxkxy, dxykxy_tr = svgd_kernel(x, dim, h)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))

    phi = (tf.matmul(kxy, sq) + dxkxy) / n

    return ksd, phi


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample, GW1 = generator(z)

KSD, D_fake = ksd_emp(G_sample)

Loss2 = tf.reduce_sum(diag_gradient(D_fake, G_sample)) / mb_size
Loss1 = tf.trace(tf.matmul(D_fake, tf.transpose(S_q(G_sample)))) / mb_size
# Loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
Loss = Loss1 + Loss2

# range_penalty_g = 10*(generator(tf.constant(1, shape=[1, 1], dtype=tf.float32)) -
#                       generator(tf.constant(-1, shape=[1, 1], dtype=tf.float32)))
# range_penalty_g = tf.Print(range_penalty_g, [range_penalty_g], message="range_penalty_g"+"-values:")

# G_solver = (tf.train.AdamOptimizer(learning_rate=1e-2).minimize(Loss, var_list=theta_G))
# G_solver_ksd = (tf.train.AdamOptimizer(learning_rate=1e-2).minimize(KSD, var_list=theta_G))


#######################################################################################################################
#######################################################################################################################
# Tests

sess = tf.Session()
sess.run(tf.global_variables_initializer())

N = 1000
loss_lst = np.zeros(N)
ksd_lst = np.zeros(N)
for i in range(N):
    out = sess.run([Loss, KSD],
                   feed_dict={z: sample_z(mb_size, z_dim)})
    loss_lst[i] = out[0]
    ksd_lst[i] = out[1]

# out = sess.run([tf.stack([tf.gradients(G_sample[:, i], z)[0][:, i] for i in range(z_dim)], axis=0), gw1, G_sample, z],
#                feed_dict={z: sample_z(mb_size, z_dim)})

# out = sess.run([z, G_sample, GW1, S_q(z), tfgrad_phi, grad_phi, Loss1],
#                feed_dict={z: sample_z(mb_size, z_dim)})
# x, fx, dfx_l, sq, tfg_fx, g_fx, sq_fx = out

out = sess.run([G_sample, S_q(G_sample), KSD, D_fake, Loss2, Loss1, Loss, svgd_kernel(G_sample)],
               feed_dict={z: sample_z(mb_size, z_dim)})

x0, sq, ksd, phi, loss2, loss1, loss, svgd_kernel1 = out
Kxy, dxKxy, dxyKxy_tr = svgd_kernel1

n = mb_size


def offdiag_sum(mat):
    return np.sum(mat) - np.trace(mat)


def S_q_np(x):
    return np.expand_dims(np.matmul(mu1 - x, Sigma1_inv), -1)
# S_q_np(x0)


def k(x, y, h=1.):
    return np.exp(-np.sum((x-y)**2)/(2 * h**2))
# k(x[0], x[1])


def dxk(x, y, h=1.):
    return np.expand_dims(- (x-y) * k(x, y, h) / (h**2), -1)
# dxk(x[0], x[1])


def dxyk_tr(x, y, h=1., d=z_dim):
    return (- (np.sum((x-y)**2) / (h**4)) + (d / (h**2))) * k(x, y, h)
# dxyk_tr(x[0], x[1])


def dxk_sq(x, y, h=1.):
    # np.matmul(dxk(x0[1], x0[0], h).T, S_q_np(x0[0]))
    return np.matmul(dxk(y, x, h).T, S_q_np(x))
# dxk_sq(x0[0], x0[1])


# def u(x, y, h=1.):
#     t1 = k(x, y, h) * np.matmul(S_q_np(x).T, S_q_np(y))
#     t2 = np.matmul(S_q_np(x).T, dxk(y, x, h))
#     t2t = np.matmul(S_q_np(y).T, dxk(x, y, h))
#     t3 = dxyk_tr(x, y, h)
#     return t1 + t2 + t2t + t3
# u(x[0], x[1])


# u_mat = np.zeros((mb_size, mb_size))
k_mat = np.zeros((n, n))
dxk_mat = np.zeros((n, n, z_dim))
dxyk_tr_mat = np.zeros((n, n))
dxk_sq_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        # u_mat[i, j] = u(x0[i], x0[j])
        k_mat[i, j] = k(x0[i], x0[j])
        dxk_mat[i, j] = dxk(x0[i], x0[j]).reshape(z_dim)
        dxyk_tr_mat[i, j] = dxyk_tr(x0[i], x0[j])
        dxk_sq_mat[i, j] = dxk_sq(x0[i], x0[j])


loss1
np.trace(np.matmul(phi, sq.T)) / n

loss2
(- np.trace(Sigma1_inv) / n) + ((offdiag_sum(dxyk_tr_mat) + offdiag_sum(dxk_sq_mat)) / (n**2))


#######################################################################################################################
#######################################################################################################################

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# ksd_loss = np.zeros(N)
# G_loss = np.zeros(N)
# D_Loss_curr = G_Loss_curr = None
#
# for it in range(N):
#     for _ in range(n_G):
#         _, G_Loss_curr, ksd_curr = sess.run([G_solver, Loss, ksd],
#                                             feed_dict={z: sample_z(mb_size, z_dim)})
#     G_loss[it] = G_Loss_curr
#     ksd_loss[it] = ksd_curr
#
#     if np.isnan(G_Loss_curr):
#         print("G_loss:", it)
#         break
#
#     if it % 10 == 0:
#         noise = sample_z(100, 1)
#         x_range = np.reshape(np.linspace(-5, 5, 500, dtype=np.float32), newshape=[500, 1])
#         z_range = np.reshape(np.linspace(-5, 5, 500, dtype=np.float32), newshape=[500, 1])
#         samples = sess.run(generator(noise.astype(np.float32)))
#         gen_func = sess.run(generator(z_range))
#         sample_mean = np.mean(samples)
#         sample_sd = np.std(samples)
#         print(it, ":", sample_mean, sample_sd)
#         print("ksd_loss:", ksd_curr)
#         print("G_loss:", G_Loss_curr)
#         print("w:", G_W1.eval(session=sess), "b:", G_b1.eval(session=sess))
#         # plt.scatter(samples[:, 0], samples[:, 1], color='b')
#         # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
#         plt.plot()
#         # plt.subplot(212)
#         # plt.plot(x_range, disc_func)
#         plt.subplot(212)
#         plt.ylim(-5, 5)
#         plt.plot(z_range, gen_func)
#         plt.subplot(211)
#         plt.ylim(-3, 5)
#         plt.plot(range(100), samples[:, 0], 'ro', color='b', ms=1)
#         plt.axhline(mu1, color='r')
#         plt.title(
#             "iter {0:04d}, {{G: {1:.4f}, mu: {2:.4f}, sd: {3:.4f}}}".format(it, G_Loss_curr, sample_mean, sample_sd))
#         plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
#         plt.close()
#
# sess.close()
#
#
# np.savetxt(EXP_DIR + "loss_ksd.csv", ksd_loss, delimiter=",")
# plt.plot(ksd_loss)
# plt.ylim(ymin=0)
# plt.axvline(np.argmin(ksd_loss), ymax=np.min(ksd_loss), color="r")
# plt.title("KSD (min at iter {})".format(np.argmin(ksd_loss)))
# plt.savefig(EXP_DIR + "ksd.png", format="png")
# plt.close()

