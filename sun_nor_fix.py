"""

Stein_GAN: sun_nor_fix.py

Created on 8/30/18 9:03 PM

@author: Hanxi Sun

"""
# create sun shaped mixture of 9 normal distr.


ON_SERVER = False
# ON_SERVER = True

import tensorflow as tf
import numpy as np
import matplotlib
from sklearn.decomposition import PCA
import os
import sys
from datetime import datetime

if ON_SERVER:
    matplotlib.use('agg')

import matplotlib.pyplot as plt


def now_str():
    now = datetime.now().strftime('%m%d%H%M%S.%f').split('.')
    return "%s%02d" % (now[0], int(now[1]) // 10000)


if ON_SERVER:
    # z_dim = int(sys.argv[1])
    # h_dim_g = h_dim_d = int(sys.argv[2])
    ini_var = float(sys.argv[1])
    d_normal = int(sys.argv[2])
    lr_d_0 = lr_g_0 = float(sys.argv[3])
    inter_decay = int(sys.argv[4])
    md = int(sys.argv[5])  # distance between centers
    sd_normal = float(sys.argv[6])

else:
    d_normal = 20
    lr_d_0 = lr_g_0 = 1e-4
    inter_decay = 200
    ini_var = 3.
    md = 10  # distance between centers
    sd_normal = 0.5


z_dim = out_dim_g = h_dim_d = h_dim_g = 200
n_D = 5
n_G = 1
lbd_0 = 5
top = out_dim_g - d_normal

rho = .9  # shape of the "beams" at corners

DIR = "/scratch/halstead/h/hu478/Stein_GAN" + "/output/temp/" if ON_SERVER else os.getcwd() + "/output/"
ID = now_str()
EXP = "sun_nor_fix_var={}_".format(ini_var) + "d_normal={}_".format(d_normal) + "lr={}_".format(lr_d_0)\
      + "inter_decay={}_".format(inter_decay) + "sd_normal={}_".format(sd_normal) + ID
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

print("TimeStart: " + ID)

n_out = 2000
mb_size = int(n_out/top)
N = 100001


# Target
X_dim = 2    # dimension of the target distribution
if X_dim != 2:
    raise ValueError("X_dim has to be 2 in this simulation!")

mu_x = np.array([-md, 0, md])
mu_y = np.array([md, 0, -md])
mu_xv, mu_yv = np.meshgrid(mu_x, mu_y)
mu = np.stack((mu_xv.reshape(-1), mu_yv.reshape(-1))).T

Sigma = np.array([[[1,  -rho], [-rho,  1]],
                  [[1-rho, 0], [0, 1+rho]],
                  [[1,   rho], [rho,   1]],
                  [[1+rho, 0], [0, 1-rho]],
                  [[1,     0], [0,     1]],
                  [[1+rho, 0], [0, 1-rho]],
                  [[1,   rho], [rho,   1]],
                  [[1-rho, 0], [0, 1+rho]],
                  [[1,  -rho], [-rho, 1]]])

Sigma_inv = np.linalg.inv(Sigma)
Sigma_det = np.linalg.det(Sigma)

Sigma_inv_normal = tf.ones(d_normal) * (1/sd_normal)

n_comp = mu.shape[0]

p = np.ones(n_comp) / n_comp

PLT_SHOW = False


########################################################################################################################
def output_matrix(prefix, matrix):
    if type(matrix) == int or type(matrix) == float:
        return prefix + '{}'.format(matrix)
    else:
        return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "_info.txt", 'w')
info.write(now_str() + '\n\n' + "Description: " + '\n\t' + "sun shaped distribution" +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['Number of mixture components = {}'.format(n_comp),
                        'Distances between centers = {}'.format(md),
                        'Mixture weights = {}'.format(p),
                        output_matrix("List of mu's = ", mu),
                        output_matrix("List of Sigma's = ", Sigma)]) + '\n'
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'd_normal = {}'.format(d_normal),
                        'lr_d = lr_g = {}'.format(lr_d_0), 'lbd = {}'.format(lbd_0),
                        'n_D = {}'.format(n_D), 'n_G = {}'.format(n_G)]) + '\n'
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Results: \n\t")

info.close()


########################################################################################################################
# plot the contour of the mixture on the first 2 dimensions
true_size = int(n_out/X_dim) * n_comp
show_size = mb_size
label = np.random.choice(n_comp, size=true_size, p=p)[:, np.newaxis]

true_sample = np.sum(np.stack([np.random.multivariate_normal(mu[i], Sigma[i], true_size) * (label == i)
                               for i in range(n_comp)]), 0)

# pca = PCA(n_components=2)  # require X_dim > 1
# pca.fit(true_sample)
# true_sample_pca = pca.transform(true_sample)
# mu_pca = pca.transform(mu)
#
# plt.scatter(true_sample_pca[:, 0], true_sample_pca[:, 1], color='m', alpha=0.2, s=10)
# plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color="r")
# plt.title("One sample from the target distribution")
plt.scatter(true_sample[:, 0], true_sample[:, 1], color='m', alpha=0.2, s=10)
plt.scatter(mu[:, 0], mu[:, 1], color="r")
plt.title("One sample from the target distribution")
plt.axis("equal")
if PLT_SHOW:
    plt.show()
else:
    plt.savefig(EXP_DIR + "_target_sample.png", format="png")
plt.close()


# x_l = np.min(true_sample_pca[0])
# x_r = np.max(true_sample_pca[0])
x_l = np.min(true_sample[0])
x_r = np.max(true_sample[0])
x_range = np.linspace(x_l, x_r, 500, dtype=np.float32)


########################################################################################################################
# tf version of model parameters
mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)
p_tf = tf.reshape(tf.convert_to_tensor(p, dtype=tf.float32), shape=[n_comp, 1])

initializer = tf.contrib.layers.xavier_initializer()

lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])
lr_g = tf.placeholder(tf.float32, shape=[])
lr_d = tf.placeholder(tf.float32, shape=[])


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.get_variable('D_w1', [d_normal, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [h_dim_d], initializer=initializer)
D_W3 = tf.get_variable('D_w3', [h_dim_d, d_normal], dtype=tf.float32, initializer=initializer)
D_b3 = tf.get_variable('D_b3', [d_normal], initializer=initializer)

theta_D_normal = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

SD_W1 = tf.get_variable('SD_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
SD_b1 = tf.get_variable('SD_b1', [h_dim_d], initializer=initializer)
SD_W2 = tf.get_variable('SD_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
SD_b2 = tf.get_variable('SD_b2', [h_dim_d], initializer=initializer)
SD_W3 = tf.get_variable('SD_w3', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
SD_b3 = tf.get_variable('SD_b3', [X_dim], initializer=initializer)

theta_D_stein = [SD_W1, SD_W2, SD_W3, SD_b1, SD_b2, SD_b3]
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, SD_W1, SD_W2, SD_W3, SD_b1, SD_b2, SD_b3]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)

G_W2 = tf.get_variable('g_w2', [h_dim_g, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [h_dim_g], initializer=initializer)

G_W3 = tf.get_variable('g_w3', [h_dim_g, out_dim_g], dtype=tf.float32, initializer=initializer)
G_b3 = tf.get_variable('g_b3', [out_dim_g], initializer=initializer)

G_scale = tf.get_variable('g_scale', [1, out_dim_g], initializer=tf.constant_initializer(ini_var))
G_location = tf.get_variable('g_location', [1, out_dim_g], initializer=tf.constant_initializer(0.))

theta_G_all = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_scale, G_location]


def log_densities(xs):
    log_den0 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[0], Sigma_inv_tf[0]),
                                        tf.transpose(xs - mu_tf[0]))) / 2 - np.log(Sigma_det[0]) / 2
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[1], Sigma_inv_tf[1]),
                                        tf.transpose(xs - mu_tf[1]))) / 2 - np.log(Sigma_det[1]) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[2], Sigma_inv_tf[2]),
                                        tf.transpose(xs - mu_tf[2]))) / 2 - np.log(Sigma_det[2]) / 2
    log_den3 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[3], Sigma_inv_tf[3]),
                                        tf.transpose(xs - mu_tf[3]))) / 2 - np.log(Sigma_det[3]) / 2
    log_den4 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[4], Sigma_inv_tf[4]),
                                        tf.transpose(xs - mu_tf[4]))) / 2 - np.log(Sigma_det[4]) / 2
    log_den5 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[5], Sigma_inv_tf[5]),
                                        tf.transpose(xs - mu_tf[5]))) / 2 - np.log(Sigma_det[5]) / 2
    log_den6 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[6], Sigma_inv_tf[6]),
                                        tf.transpose(xs - mu_tf[6]))) / 2 - np.log(Sigma_det[6]) / 2
    log_den7 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[7], Sigma_inv_tf[7]),
                                        tf.transpose(xs - mu_tf[7]))) / 2 - np.log(Sigma_det[7]) / 2
    log_den8 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[8], Sigma_inv_tf[8]),
                                        tf.transpose(xs - mu_tf[8]))) / 2 - np.log(Sigma_det[8]) / 2
    ld = tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p[0]) + log_den0,
                                                      np.log(p[1]) + log_den1,
                                                      np.log(p[2]) + log_den2,
                                                      np.log(p[3]) + log_den3,
                                                      np.log(p[4]) + log_den4,
                                                      np.log(p[5]) + log_den5,
                                                      np.log(p[6]) + log_den6,
                                                      np.log(p[7]) + log_den7,
                                                      np.log(p[8]) + log_den8], 0), 0), 1)
    return ld


def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def S_n(xs):
    return -xs/sd_normal


def sample_z(m, n, sd=10.):
    # s1 = np.random.normal(0, sd, size=[m, n])
    s1 = np.random.uniform(-sd, sd, size=[m, n])
    return s1


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def generator(z):
    # G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    # G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    # G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    # G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    # out = tf.multiply(G_h3, G_scale) + G_location

    # if force all the weights to be non negative
    G_h1 = lrelu(tf.matmul(z, G_W1) + G_b1, 0.1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h2 = lrelu(tf.matmul(G_h1, G_W2) + G_b2, 0.1)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location
    return out


def discriminator_stein(x):
    SD_h1 = tf.nn.relu(tf.matmul(x, SD_W1) + SD_b1)
    SD_h2 = tf.nn.relu(tf.matmul(SD_h1, SD_W2) + SD_b2)
    # D_h1 = tf.Print(D_h1, [D_h1], message="Discriminator-"+"D_h1"+"-values:")
    out = (tf.matmul(SD_h2, SD_W3) + SD_b3)
    # out = tf.Print(out, [out], message="Discriminator-"+"out"+"-values:")
    return out


def discriminator_normal(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    # D_h1 = tf.Print(D_h1, [D_h1], message="Discriminator-"+"D_h1"+"-values:")
    out = (tf.matmul(D_h2, D_W3) + D_b3)
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


G_sample = tf.reshape(generator(z)[:, :top], shape=[-1, X_dim])
G_normal = generator(z)[:, top:]
D_fake_stein = discriminator_stein(G_sample)
D_fake_normal = discriminator_normal(G_normal)


loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake_stein), 1), 1)
loss1_n = tf.expand_dims(tf.reduce_sum(tf.multiply(S_n(G_normal), D_fake_normal), 1), 1)

loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake_stein, G_sample), axis=1), 1)
loss2_n = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake_normal, G_normal), axis=1), 1)

Loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))
Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))

Loss_n = tf.abs(tf.reduce_mean(loss1_n + loss2_n)) - lbd * tf.reduce_mean(tf.square(D_fake_normal))


# with alpha power to the density
D_solver_a = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D_stein)

D_solver_n = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_n, var_list=theta_D_normal)

D_solver = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_n - Loss_alpha, var_list=theta_D)

G_solver_a = tf.train.RMSPropOptimizer(learning_rate=lr_g).minimize(Loss_alpha + Loss_n, var_list=theta_G_all)


#######################################################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")


alpha_1 = 0.01
lbd_1 = lbd_0
lr_g_1 = lr_g_0
lr_d_1 = lr_d_0


G_loss = np.zeros(N)
D_loss = np.zeros(N)

G_Loss_curr = D_Loss_curr = None

for it in range(N):
    for _ in range(n_D):
        _, _, D_Loss_curr = sess.run([D_solver_a, D_solver_n, Loss_alpha],
                                     feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0,
                                                alpha_power: alpha_1, lr_d: lr_d_1})
        # _, D_Loss_curr = sess.run([D_solver, Loss_alpha],
        #                           feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0,
        #                                      alpha_power: alpha_1, lr_d: lr_d_1})

    D_loss[it] = D_Loss_curr

    if np.isnan(D_Loss_curr):
        print("D_loss:", it)
        break

    # train Generator
    _, G_Loss_curr = sess.run([G_solver_a, Loss_alpha],
                              feed_dict={z: sample_z(mb_size, z_dim), lbd: lbd_0,
                                         alpha_power: alpha_1, lr_g: lr_g_1})

    G_loss[it] = G_Loss_curr

    if np.isnan(G_Loss_curr):
        print("G_loss:", it)
        break

    if it % inter_decay == 0:
        alpha_1 = np.min((alpha_1 + 0.01, 1))  # set alpha_1 = 1 would be original density

    if it % 1000 == 0:
        samples = sess.run(tf.reshape(generator(z)[:, :top], shape=[-1, X_dim]),
                           feed_dict={z: sample_z(show_size, z_dim)})
        # samples_1 = sess.run(tf.reshape(generator(z)[:, 8:10], shape=[-1, X_dim]),
        #                      feed_dict={z: sample_z(show_size * 10, z_dim)})
        samples_pca = pca.transform(samples)
        # samples_pca_1 = pca.transform(samples_1)

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
        plt.scatter(true_sample_pca[:, 0], true_sample_pca[:, 1], color='purple', alpha=0.2, s=4)
        plt.scatter(samples_pca[:, 0], samples_pca[:, 1], color='green', alpha=0.2, s=4)
        plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color='r')

        # plt.subplot(313)
        # plt.scatter(samples_pca_1[:, 0], samples_pca_1[:, 1], color='green', alpha=0.2, s=3)
        # plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color='r')

        if PLT_SHOW:
            plt.show()
        else:
            plt.savefig(EXP_DIR + "iter {0:04d}".format(it))

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


print("TimeEnds: " + now_str())

saver = tf.train.Saver()
saver.save(sess, EXP_DIR + "model.ckpt")

if ON_SERVER:
    samples = sess.run(tf.reshape(generator(z)[:, :top], shape=[-1, X_dim]),
                       feed_dict={z: sample_z(show_size, z_dim)})
    samples_pca = pca.transform(samples)

    plt.subplot(211)
    plt.title("Stein sample (green) vs true (purple) at iter {0:04d} with loss={1:.04f}".format(it, G_Loss_curr))
    num_bins = 50
    plt.hist(true_sample_pca[:, 0], num_bins, alpha=0.5, color="purple")
    plt.hist(samples_pca[:, 0], num_bins, alpha=0.5, color="green")
    plt.axvline(np.median(samples_pca[:, 0]), color='b')

    plt.subplot(212)
    plt.axis('equal')
    plt.scatter(true_sample_pca[:, 0], true_sample_pca[:, 1], color='purple', alpha=0.2, s=4)
    plt.scatter(samples_pca[:, 0], samples_pca[:, 1], color='green', alpha=0.2, s=4)
    plt.scatter(mu_pca[:, 0], mu_pca[:, 1], color='r')
    plt.savefig(DIR + EXP + "_final.png", format="png")
    plt.close()

