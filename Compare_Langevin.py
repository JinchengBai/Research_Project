"""

Stein_GAN: Compare_Langevin.py

Created on 8/2/18 7:07 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
import os
import sys

PLT_SHOW = False

DIR = os.getcwd() + "/output/"

md = 0  # distance between the means of the two mixture components
s1 = 1.
s2 = 1.
sr1 = 1.  # the ratio between the two dimensions in Sigma1
sr2 = 1.  # the ratio between the two dimensions in Sigma1
r1 = -.9  # correlation between the two dim
r2 = .9  # correlation between the two dim

eps_t = .1  # learning rate

N = 5000  # number of iteration
pct10 = N // 10
burnin = N // 2

X_dim = 2    # dimension of the target distribution
mu1, mu2 = np.zeros(X_dim), np.ones(X_dim) * np.sqrt(md**2/X_dim)
Sigma1 = s1 * np.array([[sr1, r1], [r1, 1/sr1]])
Sigma2 = s2 * np.array([[sr2, r2], [r2, 1/sr2]])

Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)

p1 = 0.5
p2 = 1 - p1


EXP = ("Compare_Langevin/" +
       "dim=2_md={0}_s1={5}_s2={6}_sr1={1}_sr2={2}_r1={3}_r2={4}_eps={7}".format(md, sr1, sr2, r1, r2, s1, s2, eps_t))
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

################################################################################################
################################################################################################


def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "_info.txt", 'w')
info.write("Description: " + '\n' +
           "\tLangevin Dynamics training"
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['p1 = {0}, p2 = {1}'.format(p1, p2),
                        'dist(mu1, mu2) = {}'.format(md),
                        'mu1 = {}'.format(mu1), output_matrix('sigma1 = ', Sigma1),
                        'mu2 = {}'.format(mu2), output_matrix('sigma2 = ', Sigma2),
                        'number of iterations = {0} with burnin = {1}'.format(N, burnin)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n')


################################################################################################
# show samples from target
show_size = 500
label = np.random.choice([0, 1], size=show_size, p=[p1, p2])

true_sample = (np.random.multivariate_normal(mu1, Sigma1, show_size) * np.expand_dims(1 - label, 1) +
               np.random.multivariate_normal(mu2, Sigma2, show_size) * np.expand_dims(label, 1))

# pca = PCA(n_components=1)
# pca.fit(true_sample)
# true_sample_pca = pca.transform(true_sample)
# mu1_pca = pca.transform(np.expand_dims(mu1, 0))
# mu2_pca = pca.transform(np.expand_dims(mu2, 0))

plt.scatter(true_sample[:, 0], true_sample[:, 1], color='m', alpha=0.2, s=10)
plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
plt.title("One sample from the target distribution")
if PLT_SHOW:
    plt.show()
else:
    plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()


#######################################################################################################################
#######################################################################################################################
# SGLD: https://github.com/blei-lab/edward/blob/master/edward/inferences/sgld.py
# example: edward/examples/normal_sgld.py

# another implementation: https://github.com/wiseodd/MCMC/blob/master/algo/sgld.py
# https://github.com/apache/incubator-mxnet/blob/master/example/bayesian-methods/sgld.ipynb

Gaussian_noise = np.random.normal(0, 1, size=[N, X_dim])

X = tf.Variable(tf.zeros([1, X_dim]))
Lang_noise = tf.placeholder(tf.float32, [1, X_dim])

mu1_tf = tf.placeholder(tf.float32, shape=[X_dim])
mu2_tf = tf.placeholder(tf.float32, shape=[X_dim])
# mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[X_dim])
# mu2_tf = tf.reshape(tf.convert_to_tensor(mu2, dtype=tf.float32), shape=[X_dim])

Sigma1_inv_tf = tf.placeholder(tf.float32, shape=[X_dim, X_dim])
Sigma2_inv_tf = tf.placeholder(tf.float32, shape=[X_dim, X_dim])

# Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[X_dim, X_dim])
# Sigma2_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32), shape=[X_dim, X_dim])


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
    _, _, s = sess.run([update_X, update_X_noise, X],
                       feed_dict={Lang_noise: np.expand_dims(Gaussian_noise[it, :], 0),
                                  mu1_tf: mu1,
                                  mu2_tf: mu2,
                                  Sigma1_inv_tf: Sigma1_inv,
                                  Sigma2_inv_tf: Sigma2_inv})
    samples[it, :] = s
    if (it+1) % pct10 == 0:
        print("{}0%".format((it+1)//pct10))
print("DONE!")

samples = samples[burnin:, ]


#######################################################################################################################
#######################################################################################################################


def rbf_kernel(x, y, h=1.):
    # x, y of shape (n, d)
    xy = np.matmul(x, y.T)
    x2 = np.sum(x**2, 1).reshape(-1, 1)
    y2 = np.sum(y**2, 1).reshape(1, -1)
    pdist = (x2 + y2) - 2*xy
    return np.exp(- pdist / h ** 2 / 2.0)  # kernel matrix


def mmd(x, y, h=1.):
    nx, ny = x.shape[0], y.shape[0]
    kxx = rbf_kernel(x, x, h=h)
    kxy = rbf_kernel(x, y, h=h)
    kyy = rbf_kernel(y, y, h=h)
    # need to do reduced log sum?
    return np.sum(kxx) / nx / (nx-1) + np.sum(kyy) / ny / (ny-1) - 2 * np.sum(kxy) / nx / ny


mmd_value = mmd(true_sample, samples)

print("mmd = {0:.04f}".format(mmd_value))

plt.title("vs true")
plt.scatter(true_sample[:, 0], true_sample[:, 1], color='m', alpha=0.2, s=10)
plt.scatter(samples[:, 0], samples[:, 1], color='b', alpha=0.1, s=10)
plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
plt.title("True (magenta) and Langevin (blue): MMD={0:.04f}".format(mmd_value))
if PLT_SHOW:
    plt.show()
else:
    plt.savefig(EXP_DIR + "_Langevin_sample.png", format="png")
plt.close()


info.write("mmd = {0:.04f}".format(mmd_value))
info.close()


