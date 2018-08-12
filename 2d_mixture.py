
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

PLT_SHOW = False

DIR = os.getcwd() + "/output/"
EXP = "2d_mixture_doubleCx-2"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 500
z_dim = 4  # we could use higher dimensions
h_dim_g = 50
h_dim_d = 50
N1, N, n_D, n_G = 100, 5000, 2, 1  # N1 is num of initial iterations to locate mean and variance
pct10 = N // 10

lr_g = 1e-2
lr_g1 = 1e-1  # learning rate for training the scale and location parameter

lr_d = 1e-3

# lr_ksd = 1e-3
lbd_0 = 0.5  # this could be tuned
alpha_0 = 0.01


md = 20  # distance between the means of the two mixture components
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
           "\n\t".join(['Number of mixture components = {}'.format(n_comp),
                        'Distances between centers = {}'.format(md),
                        'Mixture weights = {}'.format(p),
                        output_matrix("List of mu's = ", mu),
                        output_matrix("List of Sigma's = ", Sigma)]) + '\n'
           "Network Parameters: \n\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d),
                        'n_D = {}'.format(n_D), 'n_G = {}'.format(n_G)]) + '\n'
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Results: \n\t")

info.close()


################################################################################################
# plot the contour of the mixture on the first 2 dimensions
true_size = 1000
label = np.random.choice(n_comp, size=true_size, p=p)[:, np.newaxis]

true_sample = np.sum(np.stack([np.random.multivariate_normal(mu[i], Sigma[i], true_size) * (label == i)
                               for i in range(n_comp)]), 0)

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
x_range = np.linspace(x_l, x_r, 500, dtype=np.float32)

################################################################################################
# convert parameters to tf tensor
# tf version of model parameters
mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)
p_tf = tf.reshape(tf.convert_to_tensor(p, dtype=tf.float32), shape=[n_comp, 1])

initializer = tf.contrib.layers.xavier_initializer()

lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])

X = tf.placeholder(tf.float32, shape=[None, X_dim])

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
    # log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
    #                                     tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
    # log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
    #                                     tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    # return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
    #                                                     np.log(p2) + log_den2], 0), 0), 1)
    # # return log_den1
    # batch = tf.shape(x)[0]
    # x1 = tf.tile(tf.reshape(x, [-1]), [n_comp])
    # xs = tf.reshape(x1, [n_comp, -1, X_dim])
    #
    # mask = tf.reshape(tf.tile(tf.reshape(tf.eye(batch), [-1]), [n_comp]), [n_comp, batch, batch])
    # masked = (-tf.matmul(tf.matmul(xs - tf.expand_dims(mu_tf, 1), Sigma_inv_tf),
    #                      tf.transpose(xs, [0, 2, 1]) - tf.expand_dims(mu_tf, 2)) / 2)
    # ld_lst = tf.reduce_sum(tf.multiply(mask, masked), 1)
    #
    # ld = tf.expand_dims(tf.reduce_logsumexp(tf.log(p_tf) + ld_lst, 0), 1)

    log_den0 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[0], Sigma_inv_tf[0]),
                                        tf.transpose(xs - mu_tf[0]))) / 2 - np.log(Sigma_det[0]) / 2
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[1], Sigma_inv_tf[1]),
                                        tf.transpose(xs - mu_tf[1]))) / 2 - np.log(Sigma_det[1]) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[2], Sigma_inv_tf[2]),
                                        tf.transpose(xs - mu_tf[2]))) / 2 - np.log(Sigma_det[2]) / 2
    log_den3 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu_tf[3], Sigma_inv_tf[3]),
                                        tf.transpose(xs - mu_tf[3]))) / 2 - np.log(Sigma_det[3]) / 2
    ld = tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p[0]) + log_den0,
                                                      np.log(p[1]) + log_den1,
                                                      np.log(p[2]) + log_den2,
                                                      np.log(p[3]) + log_den3], 0), 0), 1)
    return ld


# Score function computed from the target distribution
def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, sd=1.):
    s1 = np.random.normal(0, sd, size=[m, n-1])
    s2 = np.random.binomial(1, 0.5, size=[m, 1])
    return np.concatenate((s1, s2), axis=1)


def generator(z):
    # G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    # G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    # G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    # G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    # out = tf.multiply(G_h3, G_scale) + G_location

    # if force all the weights to be non negative
    G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location
    return out


# output dimension of this function is X_dim
def discriminator(x):
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


G_sample = generator(z)

# decay_curr = decay_0
# G_sample = add_noisy(generator(z), decay_curr)
weights = tf.exp(log_densities(G_sample))

ksd, D_fake_ksd = ksd_emp(G_sample)
D_fake = discriminator(G_sample)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# x, f, p = sess.run([G_sample, D_fake, phi_y], feed_dict={z: sample_z(mb_size, z_dim)})
# print(f)
# print(p)

loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake, G_sample), axis=1), 1)

# Loss = tf.abs(tf.reduce_mean(loss1 + loss2))/norm_S

Loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake))

Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake))
# Loss = tf.abs(tf.reduce_sum(tf.multiply(weights, (loss1 + loss2))) / tf.reduce_sum(weights)) - \
#        lbd * tf.reduce_mean(tf.square(D_fake))
#
# Loss_alpha = tf.abs(tf.reduce_sum(tf.multiply(weights, (alpha_power * loss1 + loss2))) / tf.reduce_sum(weights)) -\
#              lbd * tf.reduce_mean(tf.square(D_fake))

D_solver = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss, var_list=theta_D))

G_solver = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss, var_list=theta_G)

# with alpha power to the density
D_solver_a = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D))

G_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G)


# for initial steps training G_scale and G_location
D_solver1 = (tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D))

G_solver1 = tf.train.GradientDescentOptimizer(learning_rate=lr_g1).minimize(Loss_alpha, var_list=theta_G1)


#######################################################################################################################
#######################################################################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")

print("The initial sample mean and std:")
initial_sample = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
print(np.mean(initial_sample, axis=0))
print(np.std(initial_sample, axis=0))

G_loss = np.zeros(N)
D_loss = np.zeros(N)

G_Loss_curr = D_Loss_curr = None

for it in range(N1):
    for _ in range(n_D):
        sess.run(D_solver1, feed_dict={z: sample_z(mb_size, z_dim), alpha_power: alpha_0, lbd: 10*lbd_0})

    sess.run(G_solver1, feed_dict={z: sample_z(mb_size, z_dim), alpha_power: alpha_0, lbd: 10*lbd_0})

print("initial steps done!")

alpha_1 = alpha_0
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
        alpha_1 = np.min((alpha_1 + 0.05, 1))  # set alpha_1 = 1 would be original density
        # lbd_1 = np.min((lbd_1 + 0.2, 10))  # this is just a random try

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


