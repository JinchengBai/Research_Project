
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.plotly as py  # tools to communicate with Plotly's server
import os
import sys


DIR = os.getcwd() + "/output/"
EXP = "1d_mixture_new"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)


# log_file = open(EXP_DIR + "_log.txt", 'wt')
# sys.stdout = log_file


mb_size = 500
X_dim = 1  # dimension of the target distribution, 3 for e.g.
z_dim = 2  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
N1, N, n_D, n_G = 100, 5000, 10, 1  # N1 is num of initial iterations to locate mean and variance

lr_g = 1e-3
lr_g1 = 1e-2  # learning rate for training the scale and location parameter

lr_d = 1e-3
# lr_ksd = 1e-3

lbd_0 = 0.5  # this could be tuned
lbd = tf.placeholder(tf.float32, shape=[])

# decay_0 = 0  # this controls the mixture with background noise, currently none
# decay = tf.placeholder(tf.float32, shape=[0])

alpha_0 = 0.05
alpha_power = tf.placeholder(tf.float32, shape=[])  # for initial iterations, power to the density to smooth the modes

mu1 = 3.
mu2 = -3.

Sigma1 = 1.
Sigma2 = 1.
Sigma1_inv = 1./Sigma1
Sigma2_inv = 1./Sigma2
Sigma1_det = 1.
Sigma2_det = 1.

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
# show samples from target
show_size = 300
label = np.random.choice([0, 1], size=show_size, p=[p1, p2])
true_sample = (np.random.normal(mu1, Sigma1, show_size) * (1 - label) +
               np.random.normal(mu2, Sigma2, show_size) * label)
plt.scatter(true_sample, np.zeros(show_size), color='b', alpha=0.2, s=10)
plt.axvline(x=mu1)
plt.axvline(x=mu2)
plt.title("One sample from the target distribution")
plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()
################################################################################################

# convert parameters to tf tensor
mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[1])
mu2_tf = tf.reshape(tf.convert_to_tensor(mu2, dtype=tf.float32), shape=[1])

Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[1, 1])
Sigma2_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32), shape=[1, 1])

X = tf.placeholder(tf.float32, shape=[None, X_dim])


initializer = tf.contrib.layers.xavier_initializer()

# noisy initialization for the generator
initializer2 = tf.truncated_normal_initializer(mean=0, stddev=10)
initializer3 = tf.random_uniform_initializer(minval=-1, maxval=1)


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

G_scale = tf.get_variable('g_scale', [1, X_dim], initializer=tf.constant_initializer(10.))
G_location = tf.get_variable('g_location', [1, X_dim], initializer=tf.constant_initializer(0.))

theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]
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
print(np.mean(initial_sample))
print(np.std(initial_sample))

# Draw score function
x_left = np.min([mu1, mu2]) - 3 * np.max([Sigma1, Sigma2])
x_right = np.max([mu1, mu2]) + 3 * np.max([Sigma1, Sigma2])
x_range = np.reshape(np.linspace(x_left, x_right, 500, dtype=np.float32), newshape=[500, 1])
score_func = sess.run(S_q(tf.convert_to_tensor(x_range)))
plt.plot(x_range, score_func, color='r')
plt.axhline(y=0)
plt.axvline(x=mu1)
plt.axvline(x=mu2)
plt.title("Score Function")
plt.savefig(EXP_DIR + "_score_function.png", format="png")
plt.close()


ksd_loss = np.zeros(N)
G_loss = np.zeros(N)
D_loss = np.zeros(N)

ksd_curr = G_Loss_curr = D_Loss_curr = None

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
    ksd_loss[it] = ksd_curr

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
        alpha_1 = np.power(alpha_1, 0.4)  # set alpha_1 = 1 would be original density
        lbd_1 = lbd_1 + 0.4  # this is just a random try
        # decay_curr = 10 * decay_0 / tf.cast((1 + it), dtype=tf.float32)

        samples, disc_func, phi_disc = sess.run([generator(z), discriminator(x_range), phi_func(x_range, G_sample)],
                                                feed_dict={z: sample_z(show_size, z_dim)})
        sample_mean = np.mean(samples)
        sample_sd = np.std(samples)
        print(it, ":", sample_mean, sample_sd)
        print("ksd_loss:", ksd_curr)
        print("G_loss:", G_Loss_curr)
        print("D_loss:", D_Loss_curr)
        # print("w:", G_W1.eval(session=sess), "b:", G_b1.eval(session=sess))
        # plt.scatter(samples[:, 0], samples[:, 1], color='b')
        # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")

        plt.plot(figsize=(100, 100))
        plt.subplot(323)
        plt.title("Histogram")
        num_bins = 50
        # the histogram of the data
        _, bins, _ = plt.hist(samples, num_bins, normed=1, facecolor='green', alpha=0.5)
        # add a 'best fit' line
        y = p1 * mlab.normpdf(bins, mu1, Sigma1) + p2 * mlab.normpdf(bins, mu2, Sigma2)
        plt.plot(bins, y, 'r--')
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

#
# log_file.close()
# sys.stdout = sys.__stdout__
