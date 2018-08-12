"""

Stein_GAN: Compare_Langevin.py

Created on 8/2/18 7:07 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
import os
import sys

########################################################################################################################
# ---------------------------------------------------- Parameters ---------------------------------------------------- #
########################################################################################################################

PLT_SHOW = False

DIR = os.getcwd() + "/output/"

# --------------------- Model Parameters --------------------- #

md = 5  # distance between the means of the two mixture components
# s1 = 1.
# s2 = 1.
# sr1 = 1.  # the ratio between the two dimensions in Sigma1
# sr2 = 1.
# r1 = -.9  # correlation between the two dim
# r2 = .9


X_dim = 2    # dimension of the target distribution

_mu0 = np.zeros(X_dim)
_mu1 = np.zeros(X_dim)
_mu1[0] = md
mu = np.stack((_mu0, _mu0, _mu1, _mu1))
# mu = np.stack((np.zeros(X_dim),
#                np.zeros(X_dim),
#                np.ones(X_dim) * np.sqrt(md**2/X_dim),
#                np.ones(X_dim) * np.sqrt(md**2/X_dim)))
# Sigma = np.stack((s1 * np.array([[sr1, r1], [r1, 1/sr1]]),
#                   s2 * np.array([[sr2, r2], [r2, 1/sr2]])))
Sigma = np.array([[[1, .9], [.9, 1]],
                  [[1, -.9], [-.9, 1]],
                  [[1, .9], [.9, 1]],
                  [[1, -.9], [-.9, 1]]])
Sigma_inv = np.linalg.inv(Sigma)
Sigma_det = np.linalg.det(Sigma)

n_comp = mu.shape[0]

p = np.ones(n_comp) / n_comp

# --------------------- Langevin Parameters --------------------- #

eps_t = .1  # learning rate

lang_N = 5000  # number of iteration
lang_pct10 = lang_N // 10
burn_in = lang_N // 2


# --------------------- SteinGAN Parameters --------------------- #

mb_size = 500
z_dim = 4  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
N1, N, n_D, n_G = 100, 5000, 10, 1  # N1 is num of initial iterations to locate mean and variance
pct10 = N // 10

lr_g_ini = 1e-2  # learning rate for training the scale and location parameter
lr_d = 1e-3
# lr_ksd = 1e-3
lbd_0 = 0.5  # this could be tuned
alpha_0 = 0.01
lr_g_0 = 1e-3

# --------------------- Output Directory --------------------- #

# EXP = ("Compare_Langevin/" +
#        "dim=2_md={0}_s1={5}_s2={6}_sr1={1}_sr2={2}_r1={3}_r2={4}_eps={7}".format(md, sr1, sr2, r1, r2, s1, s2, eps_t))
EXP = "Compare_Langevin/" + "081118-1"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)


# --------------------- Information --------------------- #

def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "_info.txt", 'w')
info.write("Description: " + '\n\t' + "Compare SteinGAN with Langevin Dynamics" +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['Number of mixture components = '.format(n_comp),
                        'Distances between centers = '.format(md),
                        'Mixture weights = {}'.format(p),
                        output_matrix("List of mu's = ", mu),
                        output_matrix("List of Sigma's = ", Sigma)]) +
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d),
                        'n_D = {}'.format(n_D), 'n_G = {}'.format(n_G)]) +
           "Langevin Parameters: \n\t" +
           "\n\t".join(['number of iterations = {}'.format(N),
                        'burn in = {}'.format(burn_in)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Results: \n")


# --------------------- True Sample --------------------- #
# True samples from target
true_size = 1000
label = np.random.choice(n_comp, size=true_size, p=p)[:, np.newaxis]

true_sample = np.sum(np.stack([np.random.multivariate_normal(mu[i], Sigma[i], true_size) * (label == i)
                               for i in range(n_comp)]), 0)
# true_sample = (np.random.multivariate_normal(mu1, Sigma1, true_size) * np.expand_dims(1 - label, 1) +
#                np.random.multivariate_normal(mu2, Sigma2, true_size) * np.expand_dims(label, 1))

# pca = PCA(n_components=1)
# pca.fit(true_sample)
# true_sample_pca = pca.transform(true_sample)
# mu1_pca = pca.transform(np.expand_dims(mu1, 0))
# mu2_pca = pca.transform(np.expand_dims(mu2, 0))
pca = PCA(n_components=2)  # require X_dim > 1
pca.fit(true_sample)
true_sample_pca = pca.transform(true_sample)
mu_pca = pca.transform(mu)

plt.scatter(true_sample_pca[:, 0], true_sample_pca[:, 1], color='m', alpha=0.2, s=10)
plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color="r")
plt.title("One sample from the target distribution")
if PLT_SHOW:
    plt.show()
else:
    plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()


x_l = np.min(true_sample_pca[0])
x_r = np.max(true_sample_pca[0])
x_range = np.stack((np.linspace(x_l, x_r, 500, dtype=np.float32), np.zeros(500)), 0).T



########################################################################################################################
# ------------------------------------------------ Langevin Dynamics ------------------------------------------------- #
########################################################################################################################

print("Langevin Dynamics: ")
# tf version of model parameters
mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)
p_tf = tf.reshape(tf.convert_to_tensor(p, dtype=tf.float32), shape=[n_comp, 1])


Gaussian_noise = np.random.normal(0, 1, size=[lang_N, X_dim])

X = tf.Variable(tf.zeros([1, X_dim]))
Lang_noise = tf.placeholder(tf.float32, [1, X_dim])

# mu1_tf = tf.placeholder(tf.float32, shape=[X_dim])
# mu2_tf = tf.placeholder(tf.float32, shape=[X_dim])
# Sigma1_inv_tf = tf.placeholder(tf.float32, shape=[X_dim, X_dim])
# Sigma2_inv_tf = tf.placeholder(tf.float32, shape=[X_dim, X_dim])

# mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[X_dim])
# mu2_tf = tf.reshape(tf.convert_to_tensor(mu2, dtype=tf.float32), shape=[X_dim])
# Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[X_dim, X_dim])
# Sigma2_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32), shape=[X_dim, X_dim])

# X = tf.get_variable('X', [1, X_dim], initializer=initializer)
initializer = tf.assign(X, np.random.normal(0, 1, size=[1, X_dim]))

# x = sample_z(1, X_dim)
# (- np.matmul(np.matmul(np.expand_dims(x - mu, 1), Sigma_inv), np.expand_dims(x - mu, 2))/2).reshape(n_comp, 1)

# log_den1 = - tf.matmul(tf.matmul(X - mu1_tf, Sigma1_inv_tf),
#                                     tf.transpose(X - mu1_tf)) / 2
# log_den2 = - tf.diag_part(tf.matmul(tf.matmul(X - mu2_tf, Sigma2_inv_tf),
#                                     tf.transpose(X - mu2_tf))) / 2
# log_den = tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
#                                                        np.log(p2) + log_den2], 0), 0), 1)

log_den_lst = tf.reshape(- tf.matmul(tf.matmul(tf.expand_dims(X - mu_tf, 1), Sigma_inv_tf),
                                     tf.expand_dims(X - mu_tf, 2)) / 2,
                         shape=[n_comp, 1])
log_den = tf.expand_dims(tf.reduce_logsumexp(tf.log(p_tf) + log_den_lst, 0), 1)

# log_den_0 = - tf.matmul(tf.matmul(X - mu_tf[0], Sigma_inv_tf[0]),
#                         tf.transpose(X - mu_tf[0])) / 2
# log_den_1 = - tf.matmul(tf.matmul(X - mu_tf[1], Sigma_inv_tf[1]),
#                         tf.transpose(X - mu_tf[1])) / 2
# log_den_2 = - tf.matmul(tf.matmul(X - mu_tf[2], Sigma_inv_tf[2]),
#                         tf.transpose(X - mu_tf[2])) / 2
# log_den_3 = - tf.matmul(tf.matmul(X - mu_tf[3], Sigma_inv_tf[3]),
#                         tf.transpose(X - mu_tf[3])) / 2
# log_den_old = tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p[0]) + log_den_0,
#                                                            np.log(p[1]) + log_den_1,
#                                                            np.log(p[2]) + log_den_2,
#                                                            np.log(p[3]) + log_den_3], 0), 0), 1)

S_q = tf.gradients(log_den, X)[0]


update_X = tf.assign(X, tf.add(X, eps_t * S_q/2.))
update_X_noise = tf.assign(X, tf.add(X, Lang_noise * np.sqrt(eps_t)))

sess = tf.Session()
sess.run(initializer)


lang_samples = np.zeros((lang_N, X_dim))

for it in range(lang_N):
    _, _, s = sess.run([update_X, update_X_noise, X],
                       feed_dict={Lang_noise: np.expand_dims(Gaussian_noise[it, :], 0)})
    lang_samples[it, :] = s
    if (it+1) % lang_pct10 == 0:
        print("{}0%".format((it+1)//lang_pct10))
print("DONE!\n\n")

lang_samples = lang_samples[burn_in:, ]

sess.close()


########################################################################################################################
# ---------------------------------------------------- Stein GAN ----------------------------------------------------- #
########################################################################################################################
tf.reset_default_graph()

print("Stein: ")

# tf version of model parameters
mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)
p_tf = tf.reshape(tf.convert_to_tensor(p, dtype=tf.float32), shape=[n_comp, 1])
X_range = tf.convert_to_tensor(x_range, dtype=tf.float32)


lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])  # for initial iterations, power to the density to smooth the modes
lr_g = tf.placeholder(tf.float32, shape=[])


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

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)

G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], initializer=initializer)

G_W3 = tf.get_variable('g_w3', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [X_dim], initializer=initializer)

G_scale = tf.get_variable('g_scale', [1, X_dim], initializer=tf.constant_initializer(30.))
G_location = tf.get_variable('g_location', [1, X_dim], initializer=tf.constant_initializer(0.))

theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]
theta_G1 = [G_scale, G_location]


# --------------------- Define Network --------------------- #
def log_densities(x):
    # x = sample_z(mb_size, X_dim)
    # x = tf.convert_to_tensor(x, dtype=tf.float32)
    batch = tf.shape(x)[0]
    x1 = tf.tile(tf.reshape(x, [-1]), [n_comp])
    xs = tf.reshape(x1, [n_comp, -1, X_dim])

    mask = tf.reshape(tf.tile(tf.reshape(tf.eye(batch), [-1]), [n_comp]), [n_comp, batch, batch])
    masked = (-tf.matmul(tf.matmul(xs - tf.expand_dims(mu_tf, 1), Sigma_inv_tf),
                         tf.transpose(xs, [0, 2, 1]) - tf.expand_dims(mu_tf, 2)) / 2)
    ld_lst = tf.reduce_sum(tf.multiply(mask, masked), 1)

    ld = tf.expand_dims(tf.reduce_logsumexp(tf.log(p_tf) + ld_lst, 0), 1)

    # log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf, Sigma_inv_tf),
    #                                     tf.transpose(xs - mu_tf))) / 2
    # log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
    #                                     tf.transpose(xs - mu2_tf))) / 2
    # return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
    #                                                     np.log(p2) + log_den2], 0), 0), 1)

    return ld


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

# ksd, D_fake_ksd = ksd_emp(G_sample)
D_fake = discriminator(G_sample)

# G_sample_fake_fake = (G_sample - tf.reduce_mean(G_sample)) * 1 + tf.reduce_mean(G_sample)
# D_fake_fake = discriminator(G_sample)


# range_penalty_g = 10*(generator(tf.constant(1, shape=[1, 1], dtype=tf.float32)) -
#                       generator(tf.constant(-1, shape=[1, 1], dtype=tf.float32)))
# range_penalty_g = tf.Print(range_penalty_g, [range_penalty_g], message="range_penalty_g"+"-values:")

# --------------------- Losses --------------------- #

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

G_solver1 = tf.train.GradientDescentOptimizer(learning_rate=lr_g_ini).minimize(Loss_alpha, var_list=theta_G1)

# G_solver_ksd = (tf.train.GradientDescentOptimizer(learning_rate=lr_ksd).minimize(ksd, var_list=theta_G))

# --------------------- Training --------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")

score_func = sess.run(S_q(X_range))
plt.plot(x_range[:, 0], score_func, color='r')
plt.axhline(y=0)
[plt.axvline(x=x) for x in mu_pca[:, 0]]
plt.title("Score Function")
if PLT_SHOW:
    plt.show()
else:
    plt.savefig(EXP_DIR + "_score_function.png", format="png")
plt.close()


G_loss = np.zeros(N)
D_loss = np.zeros(N)

G_Loss_curr = D_Loss_curr = None

for _ in range(N1):
    for _ in range(n_D):
        sess.run(D_solver1, feed_dict={z: sample_z(mb_size, z_dim), alpha_power: alpha_0, lbd: 10*lbd_0})
    sess.run(G_solver1, feed_dict={z: sample_z(mb_size, z_dim), alpha_power: alpha_0, lbd: 10*lbd_0})

print("initial steps done!")

alpha_1 = alpha_0
lbd_1 = lbd_0
lr_g_1 = lr_g_0

for it in range(N):

    for _ in range(n_D):
        _, D_Loss_curr = sess.run([D_solver_a, Loss_alpha],
                                  feed_dict={z: sample_z(mb_size, z_dim),
                                             lbd: lbd_0, alpha_power: alpha_1})

    D_loss[it] = D_Loss_curr

    if np.isnan(D_Loss_curr):
        print("D_loss:", it)
        break

    # train Generator
    _, G_Loss_curr = sess.run([G_solver_a, Loss_alpha],
                              feed_dict={z: sample_z(mb_size, z_dim),
                                         lbd: lbd_0, alpha_power: alpha_1, lr_g: lr_g_1})

    G_loss[it] = G_Loss_curr

    if np.isnan(G_Loss_curr):
        print("G_loss:", it)
        break

    if it % 50 == 0:
        alpha_1 = np.min((alpha_1 + 0.1, 1))  # set alpha_1 = 1 would be original density
        # lbd_1 = np.min((lbd_1 + 0.2, 10))  # this is just a random try

        samples, disc_func, phi_disc = sess.run([generator(z), discriminator(X_range), phi_func(X_range, G_sample)],
                                                feed_dict={z: sample_z(true_size, z_dim)})
        samples_pca = pca.transform(samples)
        print(it, "G_loss:", G_Loss_curr)
        print(it, "D_loss:", D_Loss_curr)

        plt.plot(figsize=(100, 100))
        plt.subplot(323)
        plt.title("Histogram")
        num_bins = 50
        # the histogram of the data
        _, bins, _ = plt.hist(samples_pca[:, 0], num_bins, normed=1, facecolor='green', alpha=0.5)
        # add a 'True' line
        # y = p1 * mlab.normpdf(bins, mu1, Sigma1) + p2 * mlab.normpdf(bins, mu2, Sigma2)
        true_density = gaussian_kde(data)
        plt.plot(bins, y, 'r--')
        plt.axvline(np.median(samples), color='b')
        plt.ylabel('Probability')
        # # Tweak spacing to prevent clipping of ylabel
        # plt.subplots_adjust(left=0.15)
        #
        # plot_url = py.plot_mpl(fig, filename='docs/histogram-mpl-legend')

        plt.subplot(325)
        plt.title("vs true")
        bins = np.linspace(x_left, x_right, num_bins)
        plt.hist(true_sample, bins, alpha=0.5, color="purple")
        plt.hist(samples, bins, alpha=0.5, color="green")
        plt.axvline(np.median(samples), color='b')

        plt.subplot(322)
        plt.title("Phi from ksd")
        plt.plot(x_range, phi_disc)
        plt.axhline(y=0, color="y")
        plt.axvline(mu1, color='r')
        plt.axvline(mu2, color='r')

        plt.subplot(324)
        plt.title("Discriminator")
        plt.plot(x_range, disc_func)
        plt.axhline(y=0, color="y")
        plt.axvline(mu1, color='r')
        plt.axvline(mu2, color='r')

        plt.subplot(321)
        plt.title("Samples")
        plt.scatter(true_sample, np.ones(show_size), color='purple', alpha=0.2, s=10)
        plt.scatter(samples[:, 0], np.zeros(show_size), color='b', alpha=0.2, s=10)
        # plt.plot(samples[:, 0], np.zeros(100), 'ro', color='b', ms=1)
        plt.axvline(mu1, color='r')
        plt.axvline(mu2, color='r')
        plt.title(
            "iter {0:04d}, {{G: {1:.4f}, ksd: {2:.4f}}}".format(it, G_Loss_curr, ksd_curr))
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()

sess.close()


np.savetxt(EXP_DIR + "_loss_ksd.csv", ksd_loss, delimiter=",")
plt.plot(ksd_loss)
plt.ylim(ymin=0)
plt.axvline(np.argmin(ksd_loss), ymax=np.min(ksd_loss), color="r")
plt.title("KSD (min at iter {})".format(np.argmin(ksd_loss)))
plt.savefig(EXP_DIR + "_ksd.png", format="png")
plt.close()

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





########################################################################################################################
# -------------------------------------------------- MMD Evaluation -------------------------------------------------- #
########################################################################################################################

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






# mmd_value = mmd(true_sample, samples)
#
# print("mmd = {0:.04f}".format(mmd_value))
#
# plt.title("vs true")
# plt.scatter(true_sample[:, 0], true_sample[:, 1], color='m', alpha=0.2, s=10)
# plt.scatter(samples[:, 0], samples[:, 1], color='b', alpha=0.1, s=10)
# plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
# plt.title("True (magenta) and Langevin (blue): MMD={0:.04f}".format(mmd_value))
# if PLT_SHOW:
#     plt.show()
# else:
#     plt.savefig(EXP_DIR + "_Langevin_sample.png", format="png")
# plt.close()
#
#
# info.write("mmd = {0:.04f}".format(mmd_value))
# info.close()









