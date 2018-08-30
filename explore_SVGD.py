"""

Stein_GAN: explore_SVGD.py

Created on 8/20/18 4:26 PM

@author: Hanxi Sun

"""
# tf.reset_default_graph()
ON_SERVER = False
PLT_SHOW = False

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.colors as colors
from scipy.stats import laplace
import os
import sys
from datetime import datetime
if ON_SERVER:
    matplotlib.use('agg')
import matplotlib.pyplot as plt


def now_str():
    now = datetime.now().strftime('%m%d%H%M%S.%f').split('.')
    return "%s%02d" % (now[0], int(now[1]) // 10000)  # Month(2) Day(2) Hour(2) Min(2) Sec(2) MilSec(2)


DIR = "/home/sun652/Stein_GAN" + "/output/" if ON_SERVER else os.getcwd() + "/output/"
ID = now_str()
EXP = "laplace_2d1d_" + ID
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

print("TimeStart: " + ID)


# if ON_SERVER:
#     z_dim = int(sys.argv[1])
#     n_D = int(sys.argv[2])
#     md = float(sys.argv[3])
#     lr_d_0 = lr_g_0 = float(sys.argv[4])
# else:
#     z_dim = 5
#     n_D = 10
#     md = 4
#     lr_d_0 = 1e-4
#     lr_g_0 = 1e-4


mb_size = 500
# z_dim = 4  # we could use higher dimensions
# h_dim_g = 50
# h_dim_d = 50
# N = 50000
# n_D = 10
# n_G = 1
#
# lbd_0 = 0.5  # this could be tuned
# alpha_0 = 0.01
# alpha_inc = 0.02
# lr_g_0 = 1e-3
# lr_d_0 = 1e-3
lr = 0.5
N = 5000

model = laplace()

X_dim = 2    # dimension of the target distribution


# plot parameters
show_size = mb_size
bd = 5

# grids & true densities
n_bins = 100
x1lim = [-bd, bd]
x2lim = [-bd, bd]
# delta = 0.025  # grid size
# x1 = np.arange(x1lim[0], x1lim[1], delta)
# x2 = np.arange(x2lim[0], x2lim[1], delta)
# X1, X2 = np.meshgrid(x1, x2)
# den = (model.pdf(X1) * model.pdf(X2)).reshape(X1.shape)


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
           "\n\t".join(["SVGD on 2d independent laplace distribution"]) + '\n'
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim),
                        'N = {}'.format(N),
                        'lr = {}'.format(lr)]) + '\n')
info.close()


########################################################################################################################
# plot the true sample
true_size = show_size

true_sample = np.stack((model.rvs(true_size), np.zeros(true_size)), 0).T

for i in range(X_dim):
    _, bins, _ = plt.hist(true_sample[:, i], n_bins, density=True, alpha=.2, color=col[0], label="Sample")
    lx, rx = -5, 5
    plot_x = np.linspace(lx, rx, num=n_bins)
    den_1d = model.pdf(plot_x) if X_dim == 0 else 0
    plt.plot(plot_x, den_1d, c=col[0], ls='--', label="Density")
    plt.legend(loc="upper left")
    plt.title("True Samples at dim={}".format(i+1))
    plt.ylim(ymax=max(den_1d) + .2)
    plt.xlim(lx, rx)
    if PLT_SHOW:
        plt.show()
    else:
        plt.savefig(EXP_DIR + "_target_sample_dim{}.png".format(i+1), format="png")
    plt.close()

# plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
plt.axis('equal')
tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
plt.xlim(x1lim)
plt.ylim(x2lim)
plt.title("Scatterplot of True Samples")
plt.savefig(EXP_DIR + "_target_sample_{0:1d}d.png".format(X_dim), format="png")
plt.close()


########################################################################################################################
########################################################################################################################

X = tf.get_variable('X', shape=[mb_size, X_dim],
                    initializer=tf.random_uniform_initializer(minval=-8., maxval=8., dtype=tf.float32))


def svgd_kernel(x, h=1.):
    # Reference 1: https://github.com/ChunyuanLI/SVGD/blob/master/demo_svgd.ipynb
    # Reference 2: https://github.com/yc14600/svgd/blob/master/svgd.py
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reshape(tf.reduce_sum(tf.square(x), axis=1), shape=[tf.shape(x)[0], 1])
    X2 = tf.tile(X2_, [1, tf.shape(x)[0]])
    p_dist = tf.subtract(tf.add(X2, tf.transpose(X2)), 2 * XY)  # pairwise distance matrix

    kxy = tf.exp(- p_dist / h ** 2 / 2.0)  # kernel matrix

    sum_kxy = tf.expand_dims(tf.reduce_sum(kxy, axis=1), 1)
    dxkxy = tf.add(-tf.matmul(kxy, x), tf.multiply(x, sum_kxy)) / (h ** 2)  # sum_y dk(x, y)/dx

    return kxy, dxkxy


log_den = - tf.abs(X)
S_q = tf.gradients(log_den, X)[0]
kxy, dxkxy = svgd_kernel(X)

theta = (tf.matmul(kxy, S_q) + dxkxy)/mb_size
update_X = tf.assign(X, tf.add(X, lr * theta))


########################################################################################################################
########################################################################################################################

def rbf_kernel(x, y, h=1.):
    # x, y of shape (n, d)
    xy = np.matmul(x, y.T)
    x2 = np.sum(x**2, 1).reshape(-1, 1)
    y2 = np.sum(y**2, 1).reshape(1, -1)
    p_dist = (x2 + y2) - 2*xy
    return np.exp(- p_dist / h ** 2 / 2.0)  # kernel matrix


def mmd(x, y, h=1.):
    nx, ny = x.shape[0], y.shape[0]
    kxx = rbf_kernel(x, x, h=h)
    kxy = rbf_kernel(x, y, h=h)
    kyy = rbf_kernel(y, y, h=h)
    # need to do reduced log sum?
    return np.sum(kxx) / nx / (nx-1) + np.sum(kyy) / ny / (ny-1) - 2 * np.sum(kxy) / nx / ny


########################################################################################################################
########################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Training")
IT_START = 0
for it in range(N):
    _, sample = sess.run([update_X, X])

    if it % 100 == 0:
        print("\t", it)
        sample_mmd = mmd(sample, true_sample)

        # plt.contour(X1, X2, den, cmap=cmap, alpha=.5)
        plt.scatter(true_sample[:, 0], true_sample[:, 1], alpha=0.1, c=col[0], s=10)
        plt.scatter(sample[:, 0], sample[:, 1], alpha=0.1, c=col[1], s=10)
        tp = plt.scatter(x1lim[0] - 1, x2lim[0] - 1, c=col[0], s=10)
        fp = plt.scatter(x1lim[1] + 1, x2lim[1] + 1, c=col[1], s=10)
        plt.legend((tp, fp), ("True Sample", "Fake Sample"), loc="upper left")
        plt.axis('equal')
        plt.xlim(x1lim)
        plt.ylim(x2lim)
        plt.title("SVGD Samples at iter {0:04d}, mmd={1:.04f}".format(it + IT_START, sample_mmd))
        if PLT_SHOW:
            plt.show()
        else:
            plt.savefig(EXP_DIR + "iter {0:04d}.png".format(it + IT_START), format="png")
        plt.close()

print("TimeEnds: " + now_str())


sess.close()

