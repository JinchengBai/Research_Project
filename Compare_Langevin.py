"""

Stein_GAN: Compare_Langevin.py

Created on 8/2/18 7:07 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm
import os
import sys

DIR = os.getcwd() + "/output/"
EXP = "Compare_Langevin/" + "test_run"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

X_dim = 10    # dimension of the target distribution
mu_dist = 10  # distance between the means of the two mixture components
N = 5000
pct10 = N // 10

# z_dim = X_dim # we could use higher dimensions
# h_dim_g = 50
# h_dim_d = 50
# N1, N, n_D, n_G = 100, 5000, 10, 1  # N1 is num of initial iterations to locate mean and variance
# pct10 = N // 10
#
# lr_g = 1e-3
# lr_g1_0 = 1e-1
# lr_g1 = tf.placeholder(tf.float32, shape=[])  # learning rate for training the scale and location parameter

mu1, mu2 = np.zeros(X_dim), np.ones(X_dim) * np.sqrt(mu_dist**2/X_dim)
Sigma1 = Sigma2 = np.identity(X_dim)
mu1_pca = mu2_pca = None

Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)

p1 = 0.7
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
           "Langevin Dynamics training"
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['p1 = {0}, p2 = {1}'.format(p1, p2),
                        'mu1 = {}'.format(mu1), output_matrix('sigma1 = ', Sigma1),
                        'mu2 = {}'.format(mu2), output_matrix('sigma1 = ', Sigma2)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n')  # +
           # "Network Parameters: \n\t" +
           # "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
           #              'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d), 'n_D = {}'.format(n_D),
           #              'n_G = {}'.format(n_G)]) +
           # '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           # "Training iter: \n\t" +
           # "\n\t".join(['n_D = {}'.format(n_D), 'n_G = {}'.format(n_G)]) + '\n')
info.close()


################################################################################################
# show samples from target
show_size = 300
label = np.random.choice([0, 1], size=show_size, p=[p1, p2])

if X_dim > 1:
    true_sample = (np.random.multivariate_normal(mu1, Sigma1, show_size) * np.expand_dims(1 - label, 1) +
                   np.random.multivariate_normal(mu2, Sigma2, show_size) * np.expand_dims(label, 1))

    pca = PCA(n_components=1)
    pca.fit(true_sample)
    true_sample_pca = pca.transform(true_sample)
    mu1_pca = pca.transform(np.expand_dims(mu1, 0))
    mu2_pca = pca.transform(np.expand_dims(mu2, 0))
else:
    true_sample_pca = (np.random.normal(mu1, Sigma1, show_size) * (1 - label) +
                       np.random.normal(mu2, Sigma2, show_size) * label)
    mu1_pca, mu2_pca = mu1, mu2

print("True sample pca mean = {0:.04f}, std = {1:.04f}".format(np.mean(true_sample_pca), np.std(true_sample_pca)))
plt.scatter(true_sample_pca, np.zeros(show_size), color='b', alpha=0.2, s=10)
plt.axvline(x=mu1_pca)
plt.axvline(x=mu2_pca)
plt.title("One sample from the target distribution")
plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()

x_left = np.min([mu1_pca, mu2_pca]) - 3 * np.max([Sigma1, Sigma2])
x_right = np.max([mu1_pca, mu2_pca]) + 3 * np.max([Sigma1, Sigma2])
x_range = np.reshape(np.linspace(x_left, x_right, 500, dtype=np.float32), newshape=[500, 1])


#######################################################################################################################
#######################################################################################################################
# SGLD: https://github.com/blei-lab/edward/blob/master/edward/inferences/sgld.py
# example: edward/examples/normal_sgld.py

# another implementation: https://github.com/wiseodd/MCMC/blob/master/algo/sgld.py
# https://github.com/apache/incubator-mxnet/blob/master/example/bayesian-methods/sgld.ipynb

# ------------------------ parameter -------------------- #
eps_t = 1
Gaussian_noise = np.random.normal(0, 1, size=[N, X_dim])

X = tf.Variable(tf.zeros([1, X_dim]))
Lang_noise = tf.placeholder(tf.float32, [1, X_dim])

mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[X_dim])
mu2_tf = tf.reshape(tf.convert_to_tensor(mu2, dtype=tf.float32), shape=[X_dim])

Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[X_dim, X_dim])
Sigma2_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32), shape=[X_dim, X_dim])


# X = tf.get_variable('X', [1, X_dim], initializer=initializer)
initializer = tf.assign(X, np.random.normal(0, 1, size=[1, X_dim]))

log_den1 = - tf.diag_part(tf.matmul(tf.matmul(X - mu1_tf, Sigma1_inv_tf),
                                    tf.transpose(X - mu1_tf))) / 2
log_den2 = - tf.diag_part(tf.matmul(tf.matmul(X - mu2_tf, Sigma2_inv_tf),
                                    tf.transpose(X - mu2_tf))) / 2

log_den = tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                       np.log(p2) + log_den2], 0), 0), 1)
S_q = tf.gradients(log_den, X)[0]

update_X = tf.assign(X, tf.add(X, eps_t*S_q/2.))
update_X_noise = tf.assign(X, tf.add(X, Lang_noise * np.sqrt(eps_t)))

sess = tf.Session()
sess.run(initializer)

samples = np.zeros((N, X_dim))

for it in range(N):
    iti = sess.run([update_X, update_X_noise, X],
                   feed_dict={Lang_noise: np.expand_dims(Gaussian_noise[it, :], 0)})
    samples[it, :] = iti[2]


num_bins = 50
plt.title("vs true")
# bins = np.linspace(x_left, x_right, num_bins)
plt.hist(true_sample_pca, num_bins, alpha=0.5, color="purple", density=True)
plt.hist(pca.transform(samples[2500:, :]), num_bins, alpha=0.5, color="green", density=True)
plt.show()






