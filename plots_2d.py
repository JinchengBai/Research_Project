"""

Stein_GAN: plots_2d.py

Created on 8/15/18 10:05 PM

@author: Hanxi Sun

"""

ON_SERVER = False
# ON_SERVER = True

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.colors as colors
from scipy.stats import multivariate_normal
import os
import sys
from datetime import datetime


if ON_SERVER:
    matplotlib.use('agg')

import matplotlib.pyplot as plt


def now_str():
    now = datetime.now().strftime('%m%d%H%M%S.%f').split('.')
    return "%s%02d" % (now[0], int(now[1]) // 10000)


DIR = "/home/sun652/Stein_GAN" + "/output/" if ON_SERVER else os.getcwd() + "/output/"
ID = now_str()
EXP = "plots_2d_mixture_" + ID
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

print("TimeStart: " + ID)


if ON_SERVER:
    z_dim = int(sys.argv[1])
    n_D = int(sys.argv[2])
    md = float(sys.argv[3])
    lr_d_0 = lr_g_0 = float(sys.argv[4])
else:
    z_dim = 5
    n_D = 10
    md = 4
    lr_d_0 = 1e-4
    lr_g_0 = 1e-4


mb_size = 500
# z_dim = 8  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
N = 50000
# n_D = 10
n_G = 1

lbd_0 = 0.5  # this could be tuned
alpha_0 = 0.01
alpha_inc = 0.02
# lr_g_0 = 1e-3
# lr_d_0 = 1e-3


X_dim = 2    # dimension of the target distribution

# md = 4.
mu1, mu2 = np.zeros(X_dim), np.zeros(X_dim)
mu2[0] = md

Sigma1, Sigma2 = np.identity(X_dim), np.identity(X_dim)
Sigma1_inv, Sigma2_inv = np.linalg.inv(Sigma1), np.linalg.inv(Sigma2)
Sigma1_det, Sigma2_det = np.linalg.det(Sigma1), np.linalg.det(Sigma2)

p1 = 0.5
p2 = 1 - p1

n_comp = 2


# plot parameters
show_size = 5000
nsd = 7  # n sd for plot
plot_inits = [1, 100, 250, 500, 750]

# grids & true densities
delta = 0.025  # grid size
x1lim = [mu1[0] - nsd * Sigma1[0, 0], mu2[0] + nsd * Sigma2[0, 0]]
x2lim = [mu1[1] - nsd * Sigma1[0, 0], mu2[1] + nsd * Sigma2[0, 0]]
x1 = np.arange(x1lim[0], x1lim[1], delta)
x2 = np.arange(x2lim[0], x2lim[1], delta)
X1, X2 = np.meshgrid(x1, x2)
pos = np.stack((X1, X2), 2)
den = p1 * multivariate_normal.pdf(pos, mu1, Sigma1) + p2 * multivariate_normal.pdf(pos, mu2, Sigma2)


def truncate_colormap(col_map, min_val=0.0, max_val=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=col_map.name, a=min_val, b=max_val),
        col_map(np.linspace(min_val, max_val, n)))
    return new_cmap


# https://matplotlib.org/examples/color/colormaps_reference.html
# https://matplotlib.org/api/pyplot_summary.html?highlight=colormaps#matplotlib.pyplot.colormaps
cmap = truncate_colormap(plt.get_cmap('YlOrRd_r'), 0.2, 1)  # contour color scheme
col = ['m', 'b']  # true sample color, fake sample color


########################################################################################################################
########################################################################################################################


def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "_info.txt", 'w')
info.write("Description: " + '\n\t' +
           "\n\t".join(["RMSProp",
                        "No G1 initialize step",
                        "Add dropout in D",
                        "alpha increases {} / 100 steps".format(alpha_inc),
                        "plot every 1000 iters"]) + '\n' 
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['p1 = {0}, p2 = {1}'.format(p1, p2),
                        'mu1 = {}'.format(mu1), 'mu2 = {}'.format(mu2),
                        output_matrix('sigma1 = ', Sigma1), output_matrix('sigma2 = ', Sigma2)]) + '\n\n'
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d),
                        'N = {}'.format(N),
                        'n_D = {}'.format(n_D), 'n_G = {}'.format(n_G),
                        'lr_g_0 = {0}, lr_d_0 = {1}'.format(lr_g_0, lr_d_0),
                        'lbd_0 = {0}, alpha_0 = {1}'.format(lbd_0, alpha_0)]) + '\n')
info.close()


########################################################################################################################
# plot the contour of the mixture on the first 2 dimensions
true_size = show_size
label = np.random.choice(n_comp, size=true_size, p=[p1, p2])[:, np.newaxis]

true_sample = (np.random.multivariate_normal(mu1, Sigma1, true_size) * (1 - label) +
               np.random.multivariate_normal(mu2, Sigma2, true_size) * label)

plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
plt.axis('equal')
tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
plt.xlim(x1lim)
plt.ylim(x2lim)
plt.title("Scatterplot of True Samples")
plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()


########################################################################################################################
########################################################################################################################
# tf version of model parameters
mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[X_dim])
mu2_tf = tf.reshape(tf.convert_to_tensor(mu2, dtype=tf.float32), shape=[X_dim])

Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[X_dim, X_dim])
Sigma2_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32), shape=[X_dim, X_dim])

# initializer
initializer = tf.contrib.layers.xavier_initializer()

# tuning parameters
lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])
lr_g = tf.placeholder(tf.float32, shape=[])
lr_d = tf.placeholder(tf.float32, shape=[])

# network parameters
X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [h_dim_d], initializer=initializer)
D_W3 = tf.get_variable('D_w3', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
D_b3 = tf.get_variable('D_b3', [X_dim], initializer=initializer)

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


def sample_z(m, n, sd=1.):
    s1 = np.random.normal(0, sd, size=[m, n])
    return s1


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)

G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], initializer=initializer)

G_W3 = tf.get_variable('g_w3', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [X_dim], initializer=initializer)

G_scale = tf.get_variable('g_scale', [1, X_dim], initializer=tf.constant_initializer(10.))
G_location = tf.get_variable('g_location', [1, X_dim], initializer=tf.constant_initializer(0.))

theta_G = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_scale, G_location]


# add saver
saver = tf.train.Saver()


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location
    return out


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h1 = tf.nn.dropout(D_h1, keep_prob=0.8)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h2 = tf.nn.dropout(D_h2, keep_prob=0.8)
    out = (tf.matmul(D_h2, D_W3) + D_b3)
    return out


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)
D_fake = discriminator(G_sample)


########################################################################################################################

def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                        np.log(p2) + log_den2], 0), 0), 1)


# Score function computed from the target distribution
def S_q(xs):
    return tf.gradients(log_densities(xs), xs)[0]


# losses & solvers
loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake, G_sample), axis=1), 1)


# with alpha power to the density
Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake))

# training
# D_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D)
# G_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G)
D_solver_a = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D)
G_solver_a = tf.train.RMSPropOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G)


#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")


G_loss = np.zeros(N)
D_loss = np.zeros(N)
G_Loss_curr = D_Loss_curr = None


# parameter decay
alpha_1 = alpha_0
lbd_1 = lbd_0
lr_g_1 = lr_g_0
lr_d_1 = lr_d_0

for it in range(N):
    for _ in range(n_D):
        _, D_Loss_curr = sess.run([D_solver_a, Loss_alpha],
                                  feed_dict={z: sample_z(mb_size, z_dim),
                                             lbd: lbd_0, alpha_power: alpha_1, lr_d: lr_d_1})
    if np.isnan(D_Loss_curr):
        print("[NAN] D_loss:", it)
        break
    D_loss[it] = D_Loss_curr

    # train Generator
    _, G_Loss_curr = sess.run([G_solver_a, Loss_alpha],
                              feed_dict={z: sample_z(mb_size, z_dim),
                                         lbd: lbd_0, alpha_power: alpha_1, lr_g: lr_g_1})
    if np.isnan(G_Loss_curr):
        print("[NAN] G_loss:", it)
        break
    G_loss[it] = G_Loss_curr

    # parameter decay
    if (it+1) % 100 == 0:
        alpha_1 = min(alpha_1 + alpha_inc, 1)  # set alpha_1 = 1 would be original density
        lr_g_1 = lr_g_1
        lr_d_1 = lr_d_1

    # plots
    if (it+1) % 1000 == 0:
        fake_sample = sess.run(generator(z), feed_dict={z: sample_z(show_size, z_dim)})

        print(it+1, "G_loss:", G_Loss_curr)
        print(it+1, "D_loss:", D_Loss_curr)

        # plot: t+c
        plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
        plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
        plt.scatter(fake_sample[:, 0], fake_sample[:, 1], alpha=0.1, c=col[1], s=10)
        tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
        fp = plt.scatter(x1lim[1] + 1, x2lim[1] + 1, c=col[1], s=10)
        plt.legend((tp, fp), ("True Sample", "Fake Sample"), loc="upper left")
        plt.axis('equal')
        plt.xlim(x1lim)
        plt.ylim(x2lim)
        plt.title("Scatterplot of Generated Samples vs. True Samples at iter {0:05d}".format(it+1))
        plt.savefig(EXP_DIR + "t+c_iter{0:05d}.png".format(it+1))
        plt.close()

        # # plot: t
        # plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
        # plt.scatter(fake_sample[:, 0], fake_sample[:, 1], alpha=0.1, c=col[1], s=10)
        # tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
        # fp = plt.scatter(x1lim[1] + 1, x2lim[1] + 1, c=col[1], s=10)
        # plt.legend((tp, fp), ("True Sample", "Fake Sample"), loc="upper left")
        # plt.axis('equal')
        # plt.xlim(x1lim)
        # plt.ylim(x2lim)
        # plt.title("Scatterplot of Generated Samples vs. True Samples at iter {0:05d}".format(it+1))
        # plt.savefig(EXP_DIR + "t_iter{0:05d}.png".format(it+1))
        # plt.close()
        #
        # # plot: c
        # plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
        # plt.scatter(fake_sample[:, 0], fake_sample[:, 1], alpha=0.1, c=col[1], s=10)
        # fp = plt.scatter(x1lim[1] + 1, x2lim[1] + 1, c=col[1], s=10)
        # plt.axis('equal')
        # plt.xlim(x1lim)
        # plt.ylim(x2lim)
        # plt.title("Scatterplot of Generated Samples & True Contour at iter {0:05d}".format(it+1))
        # plt.savefig(EXP_DIR + "c_iter{0:05d}.png".format(it+1))
        # plt.close()


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

save_path = saver.save(sess, EXP_DIR + "model.ckpt")
print("Model saved in path: %s" % save_path)

print("TimeEnds: " + now_str())

if ON_SERVER:
    # save the last sample out for comparison
    fake_sample = sess.run(generator(z), feed_dict={z: sample_z(show_size, z_dim)})
    plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
    plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
    plt.scatter(fake_sample[:, 0], fake_sample[:, 1], alpha=0.1, c=col[1], s=10)
    tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
    fp = plt.scatter(x1lim[1] + 1, x2lim[1] + 1, c=col[1], s=10)
    plt.legend((tp, fp), ("True Sample", "Fake Sample"), loc="upper left")
    plt.axis('equal')
    plt.xlim(x1lim)
    plt.ylim(x2lim)
    plt.title("Scatterplot of Generated Samples vs. True Samples at iter {0:05d}".format(N))
    plt.savefig(DIR + EXP + "_final.png")
    plt.close()

    sess.close()


# restore
# run model parameters first
# sess = tf.Session()
# saver.restore(sess, EXP_DIR + "model.ckpt")

# fake_sample = sess.run(generator(z), feed_dict={z: sample_z(show_size, z_dim)})
# plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
# plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
# plt.scatter(fake_sample[:, 0], fake_sample[:, 1], alpha=0.1, c=col[1], s=10)
# tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
# fp = plt.scatter(x1lim[1] + 1, x2lim[1] + 1, c=col[1], s=10)
# plt.legend((tp, fp), ("True Sample", "Fake Sample"), loc="upper left")
# plt.axis('equal')
# plt.xlim(x1lim)
# plt.ylim(x2lim)
# plt.title("Scatterplot of Generated Samples vs. True Samples at iter {0:05d}".format(N))
# plt.show()


# from scipy.stats import norm
# mu1, mu2 = -5, 5
# Sigma1 = Sigma2 = 1.
# nsd = 8
# n_bins = 100
# p = .1
# lx, rx = mu1 - nsd * Sigma1, mu2 + nsd * Sigma2
# plot_x = np.linspace(lx, rx, num=n_bins)
# den1 = p * norm.pdf(plot_x, mu1, Sigma1) + (1-p) * norm.pdf(plot_x, mu2, Sigma2)
# den2 = (1-p) * norm.pdf(plot_x, mu1, Sigma1) + p * norm.pdf(plot_x, mu2, Sigma2)
# plt.ylim(ymax=max(den2) + 0.15)
# # plt.axis("equal")
# plt.plot(plot_x, den1, 'r--', label="0.1 N(-5, 1) + 0.9 N(5, 1)")
# plt.plot(plot_x, den2, 'b--', label="0.9 N(-5, 1) + 0.1 N(5, 1)")
# plt.legend(loc="upper left")
# plt.title("Densities")
# # plt.show()
# plt.savefig("den.png")
