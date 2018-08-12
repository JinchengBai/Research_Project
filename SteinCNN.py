"""

Stein_GAN: SteinCNN.py

Created on 8/7/18 3:05 PM

@author: Hanxi Sun

"""

import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.mlab as mlab
# import matplotlib
# matplotlib.use('agg')  # if need to run on linux servers
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import plotly.plotly as py  # tools to communicate with Plotly's server
import os
import sys

"""
Implementing CNN on z dimensions. 

"""
# kernel in the space of z instead of only on dimensions of z?

DIR = os.getcwd() + "/output/"
EXP = "CNN"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)


# log_file = open(EXP_DIR + "_log.txt", 'wt')
# sys.stdout = log_file


mb_size = 500
mu_dist = 10
X_dim = 2  # dimension of the target distribution, 3 for e.g.
z_dim = 20  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
N1, N, n_D, n_G = 100, 5000, 10, 1  # N1 is num of initial iterations to locate mean and variance
fil_size, fil_nb, fil_strd = 5, 5, 1  # CNN: filter size, number of filters, stride size
mp_k = 2  # CNN: max pooling kernel size
cnn_out_size = int(np.ceil(z_dim/mp_k) * fil_nb)


lr_g = 1e-3
lr_g1 = 1e-2  # learning rate for training the scale and location parameter

lr_d = 1e-3
# lr_ksd = 1e-3

lbd_0 = 0.5  # this could be tuned

alpha_0 = 0.01

mu1, mu2 = np.zeros(X_dim), np.ones(X_dim) * np.sqrt(mu_dist**2/X_dim)
Sigma1 = Sigma2 = np.identity(X_dim)
mu1_pca = mu2_pca = None

Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)

p1 = 0.5
p2 = 1 - p1

################################################################################################
################################################################################################


def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "_info.txt", 'w')
info.write("Description: " + '\n' +
           "KSD training"
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['mu1 = {}'.format(mu1), output_matrix('sigma1 = ', Sigma1)]) +
           "\n\t".join(['mu2 = {}'.format(mu2), output_matrix('sigma2 = ', Sigma2)]) +
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
#
#
# ################################################################################################
# # show samples from target
# show_size = 500
# label = np.random.choice([0, 1], size=(show_size, 1), p=[p1, p2])
#
# true_sample = (np.random.multivariate_normal(mu1, Sigma1, show_size) * (1 - label) +
#                np.random.multivariate_normal(mu2, Sigma2, show_size) * label)
# pca = PCA(n_components=(2 if X_dim > 1 else 1))
# pca.fit(true_sample)
# true_sample_pca = pca.transform(true_sample)
# mu1_pca = pca.transform(np.expand_dims(mu1, 0))
# mu2_pca = pca.transform(np.expand_dims(mu2, 0))
#
# # print("True sample pca mean = {0:.04f}, std = {1:.04f}".format(np.mean(true_sample_pca, 0),
# # np.std(true_sample_pca, 0)))
# plt.scatter(true_sample_pca, np.zeros(show_size), color='b', alpha=0.2, s=10)
# plt.axvline(x=mu1_pca)
# plt.axvline(x=mu2_pca)
# plt.title("One sample from the target distribution")
# plt.savefig(EXP_DIR + "_target_sample.png", format="png")
# plt.close()

################################################################################################

################################################################################################
# plot the contour of the mixture on the first 2 dimensions
X1, X2 = np.meshgrid(np.linspace(-10., 10.), np.linspace(-10., 10.))
XX = np.array([X1.ravel(), X2.ravel()]).T
Y = (p1 * np.exp(-np.diag(np.matmul(np.matmul(XX - mu1[:2], Sigma1_inv[:2, :2]), (XX - mu1[:2]).T)) / 2) +
     p2 * np.exp(-np.diag(np.matmul(np.matmul(XX - mu2[:2], Sigma2_inv[:2, :2]), (XX - mu2[:2]).T)) / 2))
Y = Y.reshape(X1.shape)
CS = plt.contour(X1, X2, Y)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.title("Contour plot of the target distribution")
plt.savefig(EXP_DIR + "_target.png", format="png")
plt.close()


# plot a real sample from the target
show_size = 500
label = np.random.choice([0, 1], size=(show_size,), p=[p1, p2])
true_sample = (np.random.multivariate_normal(mu1, Sigma1, show_size) * (1 - label).reshape((show_size, 1)) +
               np.random.multivariate_normal(mu2, Sigma2, show_size) * label.reshape((show_size, 1)))
plt.scatter(true_sample[:, 0], true_sample[:, 1], color='b', alpha=0.2, s=10)
plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
plt.title("One sample from the target distribution")
plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()

################################################################################################

lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])  # for initial iterations, power to the density to smooth the modes

# convert parameters to tf tensor
mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[X_dim])
mu2_tf = tf.reshape(tf.convert_to_tensor(mu2, dtype=tf.float32), shape=[X_dim])

Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[X_dim, X_dim])
Sigma2_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32), shape=[X_dim, X_dim])

X = tf.placeholder(tf.float32, shape=[None, X_dim])


initializer = tf.contrib.layers.xavier_initializer()

# noisy initialization for the generator
# initializer2 = tf.truncated_normal_initializer(mean=0, stddev=10)
# initializer3 = tf.random_uniform_initializer(minval=-1, maxval=1)


D_W1 = tf.get_variable('D_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [h_dim_d], initializer=initializer)
D_W3 = tf.get_variable('D_w3', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
D_b3 = tf.get_variable('D_b3', [X_dim], initializer=initializer)

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

# CNN
# ref: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
#      https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/
# raw version: convolutional_network_raw.py
# advanced version: convolutional_network.pys
# filter size: [filter_height, filter_width, in_channels, out_channels]
G_Wf = tf.get_variable('g_wf', [1, fil_size, 1, fil_nb])
G_bf = tf.get_variable('g_bf', [fil_nb], dtype=tf.float32, initializer=initializer)


# Dense Network
G_W1 = tf.get_variable('g_w1', [cnn_out_size, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)

G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], initializer=initializer)

G_W3 = tf.get_variable('g_w3', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [X_dim], initializer=initializer)

G_scale = tf.get_variable('g_scale', [1, X_dim], initializer=tf.constant_initializer(30.))
G_location = tf.get_variable('g_location', [1, X_dim], initializer=tf.constant_initializer(0.))

theta_G = [G_Wf, G_bf, G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]
theta_G1 = [G_scale, G_location]


def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                        np.log(p2) + log_den2], 0), 0), 1)


def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, sd=1.):
    s1 = np.random.normal(0, sd, size=[m, n-1])
    s2 = np.random.binomial(1, 0.5, size=[m, 1])
    return np.concatenate((s1, s2), axis=1)


def generator(z):
    # CNN
    # input: [batch, in_height, in_width, in_channels]
    z_c = tf.reshape(z, [-1, 1, z_dim, 1])
    G_c = tf.nn.conv2d(z_c, G_Wf, strides=[1, fil_strd, fil_strd, 1], padding='SAME')
    G_c = tf.nn.relu(G_c + G_bf)
    G_c = tf.nn.max_pool(G_c, ksize=[1, mp_k, mp_k, 1], strides=[1, mp_k, mp_k, 1],
                         padding='SAME')
    G_c = tf.reshape(G_c, [-1, cnn_out_size])

    # Dense Network
    G_h1 = tf.nn.relu(tf.matmul(G_c, G_W1) + G_b1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location

    # if force all the weights to be non negative
    # G_h1 = tf.nn.tanh(tf.matmul(z, tf.abs(G_W1)) + G_b1)
    # G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    # G_h2 = tf.nn.tanh(tf.matmul(G_h1, tf.abs(G_W2)) + G_b2)
    # G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    # out = tf.matmul(G_h2, tf.abs(G_W3)) + G_b3
    return out


# add background noise mixed with the generated samples
def add_noisy(g_sample, decay, bound=5):
    keep = int((1. - decay) * mb_size)
    g_sample = g_sample[0:keep, :]
    g_noise = tf.random_uniform(shape=[(mb_size - keep), X_dim], minval=-bound, maxval=bound)
    out = tf.concat([g_sample, g_noise], axis=0)
    return out


# output dimension of this function is X_dim
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h1 = tf.nn.dropout(D_h1, keep_prob=0.8)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h2 = tf.nn.dropout(D_h2, keep_prob=0.8)
    out = (tf.matmul(D_h2, D_W3) + D_b3)
    return out


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


def ksd_emp(x, n=mb_size, dim=X_dim, h=1.):
    sq = S_q(x)
    kxy, dxkxy, dxykxy_tr = svgd_kernel(x, dim, h)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    # ksd_e = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
    ksd_e = (tf.reduce_sum(t13) + t2) / (n * n)

    phi = (tf.matmul(kxy, sq) + dxkxy) / n

    return ksd_e, phi


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

    phi = (tf.matmul(kxy, S_q(x)) + dxkxy) / mb_size

    return phi


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)

# decay_curr = decay_0
# G_sample = add_noisy(generator(z), decay_curr)


ksd, D_fake_ksd = ksd_emp(G_sample)
D_fake = discriminator(G_sample)


# G_sample_fake_fake = (G_sample - tf.reduce_mean(G_sample)) * 1 + tf.reduce_mean(G_sample)
# D_fake_fake = discriminator(G_sample)


# range_penalty_g = 10*(generator(tf.constant(1, shape=[1, 1], dtype=tf.float32)) -
#                       generator(tf.constant(-1, shape=[1, 1], dtype=tf.float32)))
# range_penalty_g = tf.Print(range_penalty_g, [range_penalty_g], message="range_penalty_g"+"-values:")


loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake, G_sample), axis=1), 1)

# Loss = tf.abs(tf.reduce_mean(loss1 + loss2))/norm_S

Loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake))

Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake))


D_solver = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss, var_list=theta_D))

G_solver = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss, var_list=theta_G)

# with alpha power to the density
D_solver_a = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D))

G_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G)


# for initial steps training G_scale and G_location
D_solver1 = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D))

G_solver1 = tf.train.GradientDescentOptimizer(learning_rate=lr_g1).minimize(Loss_alpha, var_list=theta_G1)

# G_solver_ksd = (tf.train.GradientDescentOptimizer(learning_rate=lr_ksd).minimize(ksd, var_list=theta_G))


#######################################################################################################################
#######################################################################################################################


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")

print("The initial sample mean and std:")
initial_sample = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
print(np.mean(initial_sample, axis=0))
print(np.std(initial_sample, axis=0))
#
# # Draw score function
# x_left = np.min([mu1, mu2]) - 3 * np.max([Sigma1, Sigma2])
# x_right = np.max([mu1, mu2]) + 3 * np.max([Sigma1, Sigma2])
# x_range = np.reshape(np.linspace(x_left, x_right, 500, dtype=np.float32), newshape=[500, 1])
# score_func = sess.run(S_q(tf.convert_to_tensor(x_range)))
# plt.plot(x_range, score_func, color='r')
# plt.axhline(y=0)
# plt.axvline(x=mu1)
# plt.axvline(x=mu2)
# plt.title("Score Function")
# plt.savefig(EXP_DIR + "_score_function.png", format="png")
# plt.close()


# out = sess.run([generator(z)], feed_dict={z: sample_z(mb_size, z_dim)})

G_loss = np.zeros(N)
D_loss = np.zeros(N)

G_Loss_curr = D_Loss_curr = None

for it in range(N1):
    for _ in range(n_D):
        sess.run(D_solver1, feed_dict={z: sample_z(mb_size, z_dim), alpha_power: alpha_0, lbd: 10*lbd_0})

    sess.run(G_solver1, feed_dict={z: sample_z(mb_size, z_dim), alpha_power: alpha_0, lbd: 10*lbd_0})

print("initial steps done!")

alpha_1 = alpha_0
lbd_1 = lbd_0

for it in range(N):

    for _ in range(n_D):
        _, D_Loss_curr, ksd_curr = sess.run([D_solver_a, Loss_alpha, ksd],
                                            feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0, alpha_power: alpha_1})

    D_loss[it] = D_Loss_curr

    if np.isnan(D_Loss_curr):
        print("D_loss:", it)
        break

    # train Generator
    _, G_Loss_curr = sess.run([G_solver_a, Loss_alpha],
                              feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0, alpha_power: alpha_1})

    G_loss[it] = G_Loss_curr

    if np.isnan(G_Loss_curr):
        print("G_loss:", it)
        break

    if it % 50 == 0:
        print(it)
        alpha_1 = np.min((alpha_1 + 0.1, 1))  # set alpha_1 = 1 would be original density
        # lbd_1 = np.min((lbd_1 + 0.2, 10))  # this is just a random try

        samples = sess.run(generator(z), feed_dict={z: sample_z(show_size, z_dim)})
        sample_mean = np.mean(samples)
        sample_sd = np.std(samples)
        print(it, ":", sample_mean, sample_sd)
        print("G_loss:", G_Loss_curr)
        print("D_loss:", D_Loss_curr)
        # print("w:", G_W1.eval(session=sess), "b:", G_b1.eval(session=sess))
        # plt.scatter(samples[:, 0], samples[:, 1], color='b')
        # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")

        plt.plot(figsize=(100, 100))
        plt.title("Samples")
        plt.scatter(true_sample[:, 0], true_sample[:, 1], color='purple', alpha=0.2, s=10)
        plt.scatter(samples[:, 0], samples[:, 1], color='b', alpha=0.2, s=10)
        # plt.plot(samples[:, 0], np.zeros(100), 'ro', color='b', ms=1)
        plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
        plt.title(
            "iter {0:04d}, {{G: {1:.4f}, ksd: {2:.4f}}}".format(it, G_Loss_curr, ksd_curr))
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()

sess.close()


np.savetxt(EXP_DIR + "_loss_D.csv", D_loss, delimiter=",")
plt.plot(D_loss)
plt.axvline(np.argmin(D_loss), ymax=np.min(D_loss), color="r")
plt.title("loss_D (min at iter {})".format(np.argmin(D_loss)))
plt.savefig(EXP_DIR + "_loss_D.png", format="png")
plt.close()

np.savetxt(EXP_DIR + "_loss_G.csv", G_loss, delimiter=",")
plt.plot(G_loss)
plt.axvline(np.argmin(G_loss), ymax=np.min(G_loss), color="r")
plt.title("loss_G (min at iter {})".format(np.argmin(G_loss)))
plt.savefig(EXP_DIR + "_loss_G.png", format="png")
plt.close()

#
# log_file.close()
# sys.stdout = sys.__stdout__











