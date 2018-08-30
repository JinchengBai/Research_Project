"""

Stein_GAN: Compare_KSD.py

Created on 8/2/18 7:07 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

# matplotlib.use('agg')

# DIR = os.getcwd() + "/output/"
DIR = "/home/sun652/Stein_GAN" + "/output/"

# X_dim, mu_dist = 1, 6
X_dim = int(sys.argv[1])
mu_dist = float(sys.argv[2])  # distance between the means of the two mixture components


def job_str(X_dim, mu_dist):
    return "_X_dim={0:02d}_mu_dist={1:2.2f}".format(X_dim, mu_dist)


EXP = "Compare_KSD/" + "JOB" + job_str(X_dim, mu_dist)
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 500
z_dim = X_dim + 1  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
N1, N, n_D, n_G = 100, 5000, 10, 1  # N1 is num of initial iterations to locate mean and variance
pct10 = N // 10

lr_g = 1e-3
lr_g1_0 = 1e-1
lr_g1 = tf.placeholder(tf.float32, shape=[])  # learning rate for training the scale and location parameter

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
           "KSD GAN training"
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['p1 = {0}, p2 = {1}'.format(p1, p2),
                        'mu1 = {}'.format(mu1), output_matrix('sigma1 = ', Sigma1),
                        'mu2 = {}'.format(mu2), output_matrix('sigma1 = ', Sigma2)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d), 'n_D = {}'.format(n_D),
                        'n_G = {}'.format(n_G)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Training iter: \n\t" +
           "\n\t".join(['n_D = {}'.format(n_D), 'n_G = {}'.format(n_G)]) + '\n')
info.close()


################################################################################################
# show samples from target
show_size = 500
label = np.random.choice([0, 1], size=show_size, p=[p1, p2])

true_sample = (np.random.multivariate_normal(mu1, Sigma1, show_size) * np.expand_dims(1 - label, 1) +
               np.random.multivariate_normal(mu2, Sigma2, show_size) * np.expand_dims(label, 1))

pca = PCA(n_components=(2 if X_dim > 1 else 1))
pca.fit(true_sample)
true_sample_pca = pca.transform(true_sample)
mu1_pca = pca.transform(np.expand_dims(mu1, 0))
mu2_pca = pca.transform(np.expand_dims(mu2, 0))

print("True sample pca mean = {0:.04f}, std = {1:.04f}".format(np.mean(true_sample_pca, 0), np.std(true_sample_pca, 0)))
plt.scatter(true_sample_pca, np.zeros(show_size), color='b', alpha=0.2, s=10)
plt.axvline(x=mu1_pca)
plt.axvline(x=mu2_pca)
plt.title("One sample from the target distribution")
plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()


################################################################################################
################################################################################################


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


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)

G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], initializer=initializer)

G_W3 = tf.get_variable('g_w3', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [X_dim], initializer=initializer)

G_scale = tf.get_variable('g_scale', [1, X_dim], initializer=tf.constant_initializer(10.))
G_location = tf.get_variable('g_location', [1, X_dim], initializer=tf.constant_initializer(0.))

theta_G1 = [G_scale, G_location]
theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]


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
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
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
    ksd_e = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
    # ksd_e = (tf.reduce_sum(t13) + t2) / (n * n)

    phi = (tf.matmul(kxy, sq) + dxkxy) / n

    return ksd_e, phi


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)

ksd, phi = ksd_emp(G_sample)

G_solver1 = tf.train.GradientDescentOptimizer(learning_rate=lr_g1).minimize(ksd, var_list=theta_G1)
G_solver_ksd = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(ksd, var_list=theta_G)


#######################################################################################################################
#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")

for it in range(N1):
    sess.run(G_solver1, feed_dict={z: sample_z(mb_size, z_dim), lr_g1: 1})

# sess.run([G_scale, G_location])
print("initial steps done!")

ksd_loss = np.zeros(N)
ksd_curr = None

for it in range(N):
    # train Generator
    _, ksd_curr = sess.run([G_solver_ksd, ksd],
                           feed_dict={z: sample_z(mb_size, z_dim)})

    # todo: better demonstration

    ksd_loss[it] = ksd_curr

    if np.isnan(ksd_curr):
        print("***** NAN at iteration {} *****".format(it))
        break

    if it % 50 == 0:
        samples = sess.run(generator(z), feed_dict={z: sample_z(show_size, z_dim)})
        if X_dim > 1:
            samples = pca.transform(samples)

        num_bins = 50
        plt.title("vs true, ksd = {0:.04f}".format(ksd_curr))
        # bins = np.linspace(x_left, x_right, num_bins)
        plt.hist(true_sample_pca, num_bins, alpha=0.5, color="purple")
        plt.hist(samples, num_bins, alpha=0.5, color="green")
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()

    if (it + 1) % pct10 == 0:
        print("{}0%".format((it + 1) // pct10))

sess.close()


np.savetxt(EXP_DIR + "_loss_ksd.csv", ksd_loss, delimiter=",")
plt.plot(ksd_loss)
plt.ylim(ymin=0)
plt.axvline(np.argmin(ksd_loss), color="r")
plt.title("KSD (min = {1:.04f}, achieved at iter {0})".format(np.argmin(ksd_loss), np.min(ksd_loss)))
plt.savefig(EXP_DIR + "_ksd.png", format="png")
plt.close()




