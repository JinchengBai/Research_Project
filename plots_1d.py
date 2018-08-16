"""

Stein_GAN: plots_1d.py

Created on 8/15/18 9:58 PM

@author: Hanxi Sun

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from sklearn.decomposition import PCA
import os

PLT_SHOW = False

DIR = os.getcwd() + "/output/"
EXP = "plots_1d_mixture-1"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)


# plot parameters
show_size = 5000
n_bins = 100
nsd = 7  # n sd for plot


mb_size = 500
z_dim = 5  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
# N1, n_DS, N, n_D, n_G = 20000, 10000, 50000, 10, 1
N1, N, n_D, n_G = 100, 10000, 10, 1
pct10 = N // 10

lr_g = 1e-5
lr_g1 = 1e-2  # learning rate for training the scale and location parameter

lr_d = 1e-5

# lr_ksd = 1e-3
lbd_0 = 0.5  # this could be tuned
alpha_0 = 0.05

X_dim = 1    # dimension of the target distribution

mu1, mu2 = 0., 4.

Sigma1, Sigma2 = 1., 1.
Sigma1_inv, Sigma2_inv = 1./Sigma1, 1./Sigma2
Sigma1_det, Sigma2_det = 1., 1.

p1 = 0.3
p2 = 1 - p1

n_comp = 2


################################################################################################
################################################################################################


def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "_info.txt", 'w')
info.write("Description: " + '\n\t' + "Compare SteinGAN with Langevin Dynamics" +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['p1 = {0}, p2 = {1}'.format(p1, p2),
                        'mu1 = {}'.format(mu1), 'mu2 = {}'.format(mu2),
                        output_matrix('sigma1 = ', Sigma1), output_matrix('sigma2 = ', Sigma2)]) + '\n'
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d),
                        'N1 = {0}, N = {1}'.format(N1, N),
                        'n_D = {0}, n_G = {1}'.format(n_D, n_G)]) + '\n'
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n')

info.close()


################################################################################################
# plot the contour of the mixture on the first 2 dimensions
true_size = show_size
label = np.random.choice(n_comp, size=true_size, p=[p1, p2])

true_sample = (np.random.normal(mu1, Sigma1, true_size) * (1 - label) +
               np.random.normal(mu2, Sigma2, true_size) * label)


_, bins, _ = plt.hist(true_sample, n_bins, density=True, alpha=.2, color='b', label="Sample")
# bin_width = (np.max(bins) - np.min(bins)) / n_bins
# add_n_bins = int(np.ceil(3. / bin_width))
# bins = np.concatenate(([(i - add_n_bins) * bin_width + np.min(bins) for i in range(add_n_bins)],
#                        bins,
#                        [(i + 1) * bin_width + np.max(bins) for i in range(add_n_bins)]))
lx, rx = mu1 - nsd * Sigma1, mu2 + nsd * Sigma2
plot_x = np.linspace(lx, rx, num=n_bins)
den = p1 * norm.pdf(plot_x, mu1, Sigma1) + p2 * norm.pdf(plot_x, mu2, Sigma2)
plt.plot(plot_x, den, 'r--', label="Density")
plt.legend(loc="upper left")
plt.title("Histogram of True Samples")
if PLT_SHOW:
    plt.show()
else:
    plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()


################################################################################################
# convert parameters to tf tensor
# tf version of model parameters
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


def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                        np.log(p2) + log_den2], 0), 0), 1)


# Score function computed from the target distribution
def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, sd=1.):
    s1 = np.random.normal(0, sd, size=[m, n])
    return s1


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location
    return out


# output dimension of this function is X_dim
def discriminator_gan(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    # D_h1 = tf.Print(D_h1, [D_h1], message="Discriminator-"+"D_h1"+"-values:")
    out = (tf.matmul(D_h2, D_W3) + D_b3)
    # out = tf.Print(out, [out], message="Discriminator-"+"out"+"-values:")
    return out


def discriminator_stein(x):
    SD_h1 = tf.nn.relu(tf.matmul(x, SD_W1) + SD_b1)
    SD_h2 = tf.nn.relu(tf.matmul(SD_h1, SD_W2) + SD_b2)
    # D_h1 = tf.Print(D_h1, [D_h1], message="Discriminator-"+"D_h1"+"-values:")
    out = (tf.matmul(SD_h2, SD_W3) + SD_b3)
    # out = tf.Print(out, [out], message="Discriminator-"+"out"+"-values:")
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


def ksd_emp(x, n=mb_size, dim=X_dim, h=1.):  # credit goes to Hanxi!!! ;P
    sq = S_q(x)
    kxy, dxkxy, dxykxy_tr = svgd_kernel(x, dim, h)
    t13 = tf.multiply(tf.matmul(sq, tf.transpose(sq)), kxy) + dxykxy_tr
    t2 = 2 * tf.trace(tf.matmul(sq, tf.transpose(dxkxy)))
    # ksd = (tf.reduce_sum(t13) - tf.trace(t13) + t2) / (n * (n-1))
    ksd = (tf.reduce_sum(t13) + t2) / (n * n)

    phi = (tf.matmul(kxy, sq) + dxkxy) / n

    return ksd, phi


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)
D_real_gan = discriminator_gan(X)

D_fake_gan = discriminator_gan(G_sample)
D_fake_stein = discriminator_stein(G_sample)

D_loss = tf.reduce_mean(D_fake_gan) - tf.reduce_mean(D_real_gan)
G_loss = -tf.reduce_mean(D_fake_gan)

#######################################################################################################################
# clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]
alpha = tf.random_uniform(
    shape=[mb_size, 1],
    minval=0.,
    maxval=1.
)
differences = G_sample - X
interpolates = X + (alpha*differences)
gradients = tf.gradients(discriminator_gan(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
D_loss += 10 * gradient_penalty

D_solver_gan = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=theta_D_gan))
G_solver_gan = (tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G_all))

#######################################################################################################################
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# x, f, p = sess.run([G_sample, D_fake, phi_y], feed_dict={z: sample_z(mb_size, z_dim)})
# print(f)
# print(p)

loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake_stein), 1), 1)
loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake_stein, G_sample), axis=1), 1)

Loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))
Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))


D_solver = tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss, var_list=theta_D_stein)
G_solver = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss, var_list=theta_G_all)


# with alpha power to the density
D_solver_a = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D_stein))
G_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G_all)


# # for initial steps training G_scale and G_location
# D_solver1 = tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D)
# G_solver1 = tf.train.GradientDescentOptimizer(learning_rate=lr_g1).minimize(Loss_alpha, var_list=theta_G1)

#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")

# print("The initial sample mean and std:")
# initial_sample = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
# print(np.mean(initial_sample, axis=0))
# print(np.std(initial_sample, axis=0))


for it in range(N1):
    for _ in range(n_D):
        sess.run(D_solver_gan, feed_dict={z: sample_z(mb_size, z_dim), X: true_sample})

    sess.run(G_solver_gan, feed_dict={z: sample_z(mb_size, z_dim)})

    if it % 1000 == 0:
        print("initial step No.", it)
        samples = sess.run(G_sample, feed_dict={z: sample_z(true_size, z_dim)})
        samples_pca = pca.transform(samples)
        plt.subplot(211)
        plt.title("Stein sample (green) vs true (purple) at iter {0:04d}".format(it))
        num_bins = 50
        plt.hist(true_sample_pca[:, 0], num_bins, alpha=0.5, color="purple")
        plt.hist(samples_pca[:, 0], num_bins, alpha=0.5, color="green")
        plt.axvline(np.median(samples_pca[:, 0]), color='b')

        plt.subplot(212)
        plt.scatter(true_sample_pca[:, 0], true_sample_pca[:, 1], color='purple', alpha=0.2, s=10)
        plt.scatter(samples_pca[:, 0], samples_pca[:, 1], color='green', alpha=0.2, s=10)
        plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color='r')

        if PLT_SHOW:
            plt.show()
        else:
            plt.savefig(EXP_DIR + "initial_iter {0:04d}".format(it))

        plt.close()

print("initialization of Generator done!")

###################################################################################################################
# for it in range(n_DS):
#     sess.run(D_solver_a, feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0, alpha_power: alpha_1})
#
# print("initialization of Discriminator done!")

mu1_tf = tf.reshape(tf.constant([-5, 0], dtype=tf.float32), shape=[1, 2])
mu2_tf = tf.reshape(tf.constant([5, 0], dtype=tf.float32), shape=[1, 2])

Sigma0 = np.array([[1, 0], [0, 1]], dtype=np.float32)
Sigma0_inv = np.linalg.inv(Sigma0)
Sigma0_det = np.linalg.det(Sigma0)
Sigma0_inv_tf = tf.reshape(Sigma0_inv, shape=[2, 2])


def log_densities_p(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma0_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma0_det) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma0_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma0_det) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(0.5) + log_den1,
                                                        np.log(0.5) + log_den2], 0), 0), 1)


def S_p(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities_p(xs), xs)[0]


initial_f = S_q(G_sample) - S_p(G_sample)
sess.run(S_q(G_sample), feed_dict={z: sample_z(mb_size, z_dim)})
loss_f = tf.abs(D_fake_stein - initial_f)
D_solver_f = tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(loss_f, var_list=theta_D_stein)

for it in range(n_DS):
    sess.run(D_solver_f, feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0})

print("initialization of Discriminator done!")

####################################################################################################################
G_loss = np.zeros(N)
D_loss = np.zeros(N)

G_Loss_curr = D_Loss_curr = None

alpha_1 = 1.0
lbd_1 = lbd_0

for it in range(N):
    for _ in range(n_D):
        _, D_Loss_curr = sess.run([D_solver_a, Loss_alpha],
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

    if it % 100 == 0:
        # alpha_1 = np.min((alpha_1 + 0.05, 1))  # set alpha_1 = 1 would be original density

        samples = sess.run(generator(z), feed_dict={z: sample_z(true_size, z_dim)})
        samples_pca = pca.transform(samples)

        print(it, "G_loss:", G_Loss_curr)
        print(it, "D_loss:", D_Loss_curr)
        # print("w:", G_W1.eval(session=sess), "b:", G_b1.eval(session=sess))
        # plt.scatter(samples[:, 0], samples[:, 1], color='b')
        # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")

        plt.subplot(211)
        plt.title("Stein sample (green) vs true (purple) at iter {0:04d} with loss={1:.04f}".format(it, G_Loss_curr))
        num_bins = 50
        plt.hist(true_sample_pca[:, 0], num_bins, alpha=0.5, color="purple")
        plt.hist(samples_pca[:, 0], num_bins, alpha=0.5, color="green")
        plt.axvline(np.median(samples_pca[:, 0]), color='b')

        plt.subplot(212)
        plt.scatter(true_sample_pca[:, 0], true_sample_pca[:, 1], color='purple', alpha=0.2, s=10)
        plt.scatter(samples_pca[:, 0], samples_pca[:, 1], color='green', alpha=0.2, s=10)
        plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color='r')

        if PLT_SHOW:
            plt.show()
        else:
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

