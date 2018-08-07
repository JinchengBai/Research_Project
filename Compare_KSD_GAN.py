"""

Stein_GAN: Compare_KSD_GAN.py

Created on 8/2/18 7:07 PM

@author: Hanxi Sun

"""
# reference: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

DIR = os.getcwd() + "/output/"
EXP = "072118-CompareSVGD-1"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 100
X_dim = 2  # dimension of the target distribution, 3 for e.g.
z_dim = 10
h_dim_g = 50
h_dim_d = 50
N, n_D, n_G = 50000, 1, 1  # num of iterations

# As a simple example, use a 3-d Gaussian as target distribution

# parameters
p1, p2 = 0.5, 0.5
# mu1, mu2 = np.array([2, 2, 2]), np.array([-1, -1, -1])
# Sigma1 = Sigma2 = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
mu1, mu2 = np.array([2, 2]), np.array([-1, -1])
Sigma1 = Sigma2 = np.matrix([[1, 0], [0, 1]])
Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)


################################################################################################
################################################################################################


def output_matrix(prefix, matrix):
    return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "info.txt", 'w')
info.write("Compare SVGD" +
           '\n\n' + ("=" * 80 + '\n') + '\n' +
           "Description: " +
           '\n\n' + ("=" * 80 + '\n') + '\n' +
           "Model Parameters: \n\n" + "\t" +
           "\n\t".join(['p1 = {}'.format(p1), 'p2 = {}'.format(p2),
                        'mu1 = {}'.format(mu1), 'mu2 = {}'.format(mu1),
                        output_matrix('sigma1 = ', Sigma1),
                        output_matrix('sigma2 = ', Sigma2)]) +
           '\n\n' + ("=" * 80 + '\n') + '\n' +
           "Network Parameters: \n\n" + "\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d), 'n_D = {}'.format(n_D),
                        'n_G = {}'.format(n_G)]) +
           '\n\n' + ("=" * 80 + '\n') + '\n' +
           "Additional Information: \n" +
           "" + "\n")
info.close()


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
plt.savefig(EXP_DIR + "target.png", format="png")
plt.close()


# plot a real sample from the target
label = np.random.choice([0, 1], size=(mb_size,), p=[p1, p2])
sample = (np.random.multivariate_normal(mu1, Sigma1, mb_size) * (1 - label).reshape((mb_size, 1)) +
          np.random.multivariate_normal(mu2, Sigma2, mb_size) * label.reshape((mb_size, 1)))
plt.scatter(sample[:, 0], sample[:, 1], color='b', alpha=0.4, s=10)
plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
plt.title("One sample from the target distribution")
plt.savefig(EXP_DIR + "target_sample.png", format="png")
plt.close()


################################################################################################
################################################################################################

# convert parameters to tf tensor
mu1_tf = tf.convert_to_tensor(mu1, dtype=tf.float32)
mu2_tf = tf.convert_to_tensor(mu2, dtype=tf.float32)
Sigma1_inv_tf = tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32)
Sigma2_inv_tf = tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

initializer = tf.contrib.layers.xavier_initializer()

z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)
G_W2 = tf.get_variable('g_w2', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [X_dim], initializer=initializer)

theta_G = [G_W1, G_W2, G_b1, G_b2]


def svgd_kernel(x):  # adopted from reference
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reduce_sum(tf.square(x), axis=1)
    x2 = tf.reshape(X2_, shape=(tf.shape(x)[0], 1))
    X2e = tf.tile(x2, [1, tf.shape(x)[0]])
    H = tf.subtract(tf.add(X2e, tf.transpose(X2e)), 2 * XY)

    V = tf.reshape(H, [-1, 1])

    # median distance
    def get_median(v):
        v = tf.reshape(v, [-1])
        m = tf.shape(v)[0] // 2
        return tf.nn.top_k(v, m).values[m - 1]

    h = get_median(V)
    h = tf.sqrt(0.5 * h / tf.log(tf.cast(tf.shape(x)[0], tf.float32) + 1.0))

    # compute the rbf kernel
    Kxy = tf.exp(-H / h ** 2 / 2.0)

    dxkxy = -tf.matmul(Kxy, x)
    sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1)
    dxkxy = tf.add(dxkxy, tf.multiply(x, sumkxy)) / (h ** 2)


    return Kxy, dxkxy


def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
    # log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
    #                                     tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    # return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
    #                                                     np.log(p2) + log_den2], 0), 0), 1)
    return log_den1


def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, bound=1.):
    np.random.seed(1)
    return np.random.uniform(-bound, bound, size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    out = (tf.matmul(G_h1, G_W2) + G_b2)
    return out


G_sample = generator(z)
kernel_matrix, kernel_gradients = svgd_kernel(G_sample)
sq = S_q(G_sample)

phi = (tf.matmul(kernel_matrix, sq) + kernel_gradients)/mb_size

# ksd = tf.matmul(kernel_matrix, sq)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
out = sess.run([kernel_matrix, kernel_gradients, sq, phi], feed_dict={z: sample_z(mb_size, z_dim)})
[print(out_i.shape) for out_i in out]











