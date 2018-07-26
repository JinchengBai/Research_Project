"""

Stein_GAN: tests_Hanxi.py

Created on 7/16/18 8:20 PM

@author: Hanxi Sun

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.getcwd() + "/output/"
EXP = "tests"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)


########################################################################
mb_size = 100
X_dim = 3  # dimension of the target distribution, 3 for e.g.
z_dim = 5
h_dim = 20
N, each = 10000, 5  # num of iterations


########################################################################
p1 = p2 = .5
mu1, mu2 = np.array([2, 2, 2]), np.array([-1, -1, -1])
Sigma1 = Sigma2 = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)

# convert parameters to tf tensor
log_p1_tf = tf.convert_to_tensor(np.log(p1), dtype=tf.float32)
log_p2_tf = tf.convert_to_tensor(np.log(p2), dtype=tf.float32)
mu1_tf = tf.convert_to_tensor(mu1, dtype=tf.float32)
mu2_tf = tf.convert_to_tensor(mu2, dtype=tf.float32)
Sigma1_inv_tf = tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32)
Sigma2_inv_tf = tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32)


########################################################################


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


initializer = tf.contrib.layers.xavier_initializer()

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim], initializer=initializer)
G_W2 = tf.get_variable('g_w2', [h_dim, X_dim], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [X_dim], initializer=initializer)
theta_G = [G_W1, G_W2, G_b1, G_b2]

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim, X_dim], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [X_dim], initializer=initializer)
theta_D = [D_W1, D_W2, D_b1, D_b2]


def sample_z(m, n, bound=10.):
    np.random.seed(1)
    return np.random.uniform(-bound, bound, size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    out = (tf.matmul(G_h1, G_W2) + G_b2)
    return out


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.nn.tanh(tf.matmul(D_h1, D_W2) + D_b2)
    return out


G_sample = generator(z)
D_fake = discriminator(G_sample)


# log densities for a collection of samples (G_sample)
def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                        np.log(p2) + log_den2], 0), 0), 1)
    # return log_den1


# Score function computed from the target distribution
def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


xs = G_sample
log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                    tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                    tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
log_den3 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                    tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
fake_dens = tf.transpose(tf.stack([log_den1, log_den2, log_den3], 0))
fake_dens = tf.expand_dims(fake_dens, 1)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
# y, x, ys, xs = sess.run([y, x, ys, xs], feed_dict={z: sample_z(mb_size, z_dim)})
# grad, d_fake, sample = sess.run([tf.gradients(D_fake, G_sample), D_fake, G_sample],
#                                 feed_dict={z: sample_z(mb_size, z_dim)})
# grad, ys, xs = sess.run([grad, ys, xs],
#                         feed_dict={z: sample_z(mb_size, z_dim)})
# d_fake, sq = sess.run([D_fake, G_sample], feed_dict={z: sample_z(mb_size, z_dim)})
# grad, log_den, sample = sess.run([tf.gradients(log_den1, xs), log_den1, xs], feed_dict={z: sample_z(mb_size, z_dim)})
# prod, sq, grad, d_fake, sample = sess.run([tf.reduce_sum(S_q(G_sample) * D_fake, 1), S_q(G_sample),
#                                            tf.gradients(D_fake, G_sample)[0], D_fake, G_sample],
#                                           feed_dict={z: sample_z(mb_size, z_dim)})
grad, den3, den1, sample = sess.run([tf.gradients(fake_dens, xs)[0], fake_dens, log_den1, xs],
                                    feed_dict={z: sample_z(mb_size, z_dim)})
sess.close()

print(grad.shape, den3.shape, den1.shape, sample.shape)

grad_true1 = np.matmul(mu1 - sample, Sigma1_inv)
grad_true2 = np.matmul(mu2 - sample, Sigma2_inv)
grad_true3 = np.matmul(mu2 - sample, Sigma2_inv)

i = 0
grad_true = grad_true1[:, 0] + grad_true2[:, 1] + grad_true3[:, 2]
print(grad_true1[i, 0] + grad_true2[i, 1] + grad_true3[i, 2], np.sum(grad[i, ])/3, sep="\n")
print(grad_true1[i, 0], grad_true2[i, 1], grad_true3[i, 2])
print(grad[i, ])

for i in range(grad.shape[0]):
    good = True
    if np.abs(grad_true[i] - np.sum(grad[i, ])/3) > 1e-5:
        print("WRONG at {}!".format(i))
        good = False
if good:
    print("YEAH!!!")



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


# # I just simply migrated the code from WassersteinGAN here and commented out some lines we don't need.
#
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import os
#
# DIR = os.getcwd() + "/output/"
# EXP = "071618-3"
# EXP_DIR = DIR + EXP + "/"
#
# mb_size = 500
# # X_dim = 784
# X_dim = 3  # dimension of the target distribution, 3 for e.g.
# z_dim = 5
# h_dim = 20
# N, each = 10000, 5  # num of iterations
#
# # mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
# # As a simple example, use a 3-d Gaussian as target distribution
#
# # parameters
# p1 = p2 = .5
# mu1, mu2 = np.array([2, 2, 2]), np.array([-1, -1, -1])
# Sigma1 = Sigma2 = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Sigma1_inv = np.linalg.inv(Sigma1)
# Sigma2_inv = np.linalg.inv(Sigma2)
# Sigma1_det = np.linalg.det(Sigma1)
# Sigma2_det = np.linalg.det(Sigma2)
#
#
# ################################################################################################
# ################################################################################################
#
# # # plot the contour of the mixture on the first 2 dimensions
# # X1, X2 = np.meshgrid(np.linspace(-10., 10.), np.linspace(-10., 10.))
# # XX = np.array([X1.ravel(), X2.ravel()]).T
# # Y = (p1 * np.exp(-np.diag(np.matmul(np.matmul(XX - mu1[:2], Sigma1_inv[:2, :2]), (XX - mu1[:2]).T)) / 2) +
# #      p2 * np.exp(-np.diag(np.matmul(np.matmul(XX - mu2[:2], Sigma2_inv[:2, :2]), (XX - mu2[:2]).T)) / 2))
# # Y = Y.reshape(X1.shape)
# # CS = plt.contour(X1, X2, Y)
# # CB = plt.colorbar(CS, shrink=0.8, extend='both')
# # plt.title("Contour plot of the target distribution")
# # plt.savefig(EXP_DIR + "target.png", format="png")
# # plt.close()
#
#
# # convert parameters to tf tensor
# log_p1_tf = tf.convert_to_tensor(np.log(p1), dtype=tf.float32)
# log_p2_tf = tf.convert_to_tensor(np.log(p2), dtype=tf.float32)
# mu1_tf = tf.convert_to_tensor(mu1, dtype=tf.float32)
# mu2_tf = tf.convert_to_tensor(mu2, dtype=tf.float32)
# Sigma1_inv_tf = tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32)
# Sigma2_inv_tf = tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32)
# log_det1_tf = tf.convert_to_tensor(np.log(Sigma1_det), dtype=tf.float32)
# log_det2_tf = tf.convert_to_tensor(np.log(Sigma2_det), dtype=tf.float32)
#
#
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)
#
#
# X = tf.placeholder(tf.float32, shape=[None, X_dim])
#
#
# initializer = tf.contrib.layers.xavier_initializer()
#
# D_W1 = tf.get_variable('D_w1', [X_dim, h_dim], dtype=tf.float32, initializer=initializer)
# D_b1 = tf.get_variable('D_b1', [h_dim], initializer=initializer)
# D_W2 = tf.get_variable('D_w2', [h_dim, 1], dtype=tf.float32, initializer=initializer)
# D_b2 = tf.get_variable('D_b2', [1], initializer=initializer)
#
# theta_D = [D_W1, D_W2, D_b1, D_b2]
#
#
# z = tf.placeholder(tf.float32, shape=[None, z_dim])
#
# G_W1 = tf.get_variable('g_w1', [z_dim, h_dim], dtype=tf.float32, initializer=initializer)
# G_b1 = tf.get_variable('g_b1', [h_dim], initializer=initializer)
# G_W2 = tf.get_variable('g_w2', [h_dim, X_dim], dtype=tf.float32, initializer=initializer)
# G_b2 = tf.get_variable('g_b2', [X_dim], initializer=initializer)
#
# theta_G = [G_W1, G_W2, G_b1, G_b2]
#
#
# # Score function computed from the target distribution
# # def S_q(x):
# #     return tf.matmul(mu_tf - x, Sigma_inv_tf)
# # def density(x):
# #     return tf.exp(-tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf)) / 2)
#
#
# # # target log-density
# # def log_density(x):
# #     # return tf.reduce_logsumexp(-tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf))/2)
# #     # return -tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf)) / 2
# #     # return tf.log(density(x))
# #     log_den1 = -tf.matmul(tf.matmul(x - mu1_tf, Sigma1_inv_tf), tf.transpose(x - mu1_tf)) / 2
# #     log_den2 = -tf.matmul(tf.matmul(x - mu2_tf, Sigma2_inv_tf), tf.transpose(x - mu2_tf)) / 2
# #     return tf.reduce_logsumexp(tf.concat([log_p1_tf + log_den1,
# #                                           log_p2_tf + log_den2], 0))
#
#
# # log densities for a list of xs
# def log_density_lst(xs):
#     log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
#                                         tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
#     log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
#                                         tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
#     # return tf.expand_dims(tf.reduce_logsumexp(tf.stack([log_den1,
#     #                                                     tf.constant(0., shape=[mb_size])], 0), 0), 1)
#     #                                                     # np.log(p2) + log_den2], 0), 0), 1)
#     return log_den1
#
#
# # Score function computed from the target distribution
# def S_q(xs):
#     return tf.gradients(log_density_lst(xs), xs)[0]
#     # return tf.gradients(log_density(x), x)
#     # return tf.gradients(tf.log(density(x)), x)
#     # return tf.map_fn(lambda a: tf.gradients(tf.log(density(a)), a), x)
#     # return tf.matmul(mu_tf - x, Sigma_inv_tf)
#
#
# def sample_z(m, n, bound=10.):
#     np.random.seed(1)
#     return np.random.uniform(-bound, bound, size=[m, n])
#
#
# def generator(z):
#     G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
#     out = (tf.matmul(G_h1, G_W2) + G_b2)
#     return out
#
#
# def discriminator(x):
#     D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
#     out = tf.nn.tanh(tf.matmul(D_h1, D_W2) + D_b2)
#     return out
#
#
# G_sample = generator(z)
# # D_real = discriminator(X)
# D_fake = discriminator(G_sample)
#
# # D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
# # G_loss = -tf.reduce_mean(D_fake)
#
#
# Loss = tf.reduce_sum(tf.square(S_q(G_sample) * D_fake + tf.gradients(D_fake, G_sample)[0]))
#
#
# # D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
# #             .minimize(-D_loss, var_list=theta_D))
# # G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
# #             .minimize(G_loss, var_list=theta_G))
#
# D_solver = (tf.train.AdamOptimizer(learning_rate=1e-4)
#             .minimize(-Loss, var_list=theta_D))
# G_solver = (tf.train.AdamOptimizer(learning_rate=1e-4)
#             .minimize(Loss, var_list=theta_G))
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # x = tf.expand_dims(tf.convert_to_tensor(np.array([1, 1, 1])), 0)
# # sq = sess.run(S_q(x))
# # sq0 = sess.run(tf.matmul(mu_tf - x, Sigma_inv_tf))
# # print(sq, sq0)
# # sess.close()
#
#
# # if not os.path.exists('out/'):
# #     os.makedirs('out/')
# # i = 0
#
#
# # print(sess.run(tf.gradients(D_fake, G_sample), feed_dict={z: sample_z(mb_size, z_dim)}))
# # print(D_W1.eval(session=sess))
#
#
# D_loss = np.zeros(N)
# G_loss = np.zeros(N)
# for it in range(N):
#     for _ in range(each):
#         _, D_Loss_curr = sess.run([D_solver, Loss],
#                                   feed_dict={z: sample_z(mb_size, z_dim)})
#     D_loss[it] = D_Loss_curr
#
#     for _ in range(each):
#         _, G_Loss_curr = sess.run([G_solver, Loss],
#                                   feed_dict={z: sample_z(mb_size, z_dim)})
#     G_loss[it] = G_Loss_curr
#
#     if it % 100 == 0:
#         samples = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
#         print(np.mean(samples, axis=0))
#         print("D_loss", it, ":", D_Loss_curr)
#         print("G_loss", it, ":", G_Loss_curr)
#         plt.scatter(samples[:, 0], samples[:, 1], color='b')
#         plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
#         plt.title("Samples at iter {0:04d}, with loss {{D: {1:.4f}, G: {2:.4f}}}.".format(it, D_Loss_curr, G_Loss_curr))
#         plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
#         plt.close()
# sess.close()
#
#
# np.savetxt(EXP_DIR + "loss_D.csv", D_loss, delimiter=",")
# plt.plot(D_loss)
# plt.ylim(ymin=0)
# plt.axvline(np.argmin(D_loss), ymax=np.min(D_loss), color="r")
# plt.title("loss_D (min at iter {})".format(np.argmin(D_loss)))
# plt.savefig(EXP_DIR + "loss_D.png", format="png")
# plt.close()
#
# np.savetxt(EXP_DIR + "loss_G.csv", G_loss, delimiter=",")
# plt.plot(G_loss)
# plt.ylim(ymin=0)
# plt.axvline(np.argmin(G_loss), ymax=np.min(G_loss), color="r")
# plt.title("loss_G (min at iter {})".format(np.argmin(G_loss)))
# plt.savefig(EXP_DIR + "loss_G.png", format="png")
# plt.close()





######## KSD #########

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# out = sess.run([svgd_kernel(z), ksd_ty(z), ksd_emp(z)], feed_dict={z: sample_z(mb_size, z_dim)})
#
# x = sample_z(mb_size, z_dim)
# kxy, dxkxy, dxykxy_tr = out[0]
# ksd_ty, phi_ty = out[1]
# ksd_hx, phi_hx = out[2]
#
#
# def k(x, y, h=1.):
#     return np.exp(-np.sum((x-y)**2)/(2 * h**2))
#
#
# def dkdx(x, y, h=1.):
#     return - (x-y) * k(x, y, h) / (h**2)
#
#
# def tr_dkdxy(x, y, h=1., d=z_dim):
#     return (- (np.sum((x-y)**2) / (h**4)) + (d / (h**2))) * k(x, y, h)
#
#
# def sq(x):
#     return Sigma1_inv * (mu1 - x)
#
#
# def u(x, y, h=1.):
#     t1 = k(x, y, h) * np.matmul(sq(x).T, sq(y))
#     t2 = np.matmul(sq(x).T, dkdx(y, x, h))
#     t2t = np.matmul(sq(y).T, dkdx(x, y, h))
#     t3 = tr_dkdxy(x, y, h)
#     return t1, t2, t2t, t3
#
#
# t1_mat = np.zeros((mb_size, mb_size))
# t2_mat = np.zeros((mb_size, mb_size))
# t3_mat = np.zeros((mb_size, mb_size))
# u_mat = np.zeros((mb_size, mb_size))
# k_mat = np.zeros((mb_size, mb_size))
# dkdx_mat = np.zeros((mb_size, mb_size, z_dim))
# dxydkxy_tr_mat = np.zeros((mb_size, mb_size))
#
# for i in range(mb_size):
#     for j in range(mb_size):
#         t1_mat[i, j], t2_mat[i, j], t2t, t3_mat[i, j] = u(x[i], x[j])
#         u_mat[i, j] = t1_mat[i, j] + t2_mat[i, j] + t2t + t3_mat[i, j]
#         k_mat[i, j] = k(x[i], x[j])
#         dkdx_mat[i, j] = dkdx(x[i], x[j])
#         dxydkxy_tr_mat[i, j] = tr_dkdxy(x[i], x[j])
# t13_mat = t1_mat + t3_mat
#
# # print(kxy, '\n\n', k_mat)
# # print(dxkxy, '\n\n', np.sum(dkdx_mat, axis=0))
# # print(dxykxy_tr, '\n\n', tr_dkdxy_mat)
# np.sum(abs(kxy - k_mat) > 1e-5)
# np.sum(abs(dxkxy - np.sum(dkdx_mat, axis=0)) > 1e-5)
# np.sum(abs(dxykxy_tr - dxydkxy_tr_mat) > 1e-5)
#
# (np.sum(t13_mat) - np.trace(t13_mat) + 2 * np.trace(np.matmul(sq(x), dxkxy.T))) / (mb_size*(mb_size-1))
# (np.sum(u_mat) - np.trace(u_mat)) / (mb_size*(mb_size-1))
