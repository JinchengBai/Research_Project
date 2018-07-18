# I just simply migrated the code from WassersteinGAN here and commented out some lines we don't need.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

DIR = os.getcwd() + "/output/"
EXP = "071620-1"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 100
X_dim = 2  # dimension of the target distribution, 3 for e.g.
z_dim = 10
h_dim_g = 50
h_dim_d = 50
N, n_D, n_G = 50000, 1, 1  # num of iterations

# As a simple example, use a 3-d Gaussian as target distribution

# parameters
p1, p2 = 0.5, 0.5
# mu1, mu2 = np.array([2, 2, 2]), np.array([-1, -1, -1])
# Sigma1 = Sigma2 = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
mu1, mu2 = np.array([2, 2]), np.array([-1, -1])
Sigma1 = Sigma2 = np.matrix([[1, 0], [0, 1]])
Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)


################################################################################################
################################################################################################


def output_matrix(prefix, matrix):
    return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))


info = open(EXP_DIR + "info.txt", 'w')
info.write("Description: " +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\n" + "\t" +
           "\n\t".join(['p1 = {}'.format(p1), 'p2 = {}'.format(p2),
                        'mu1 = {}'.format(mu1), 'mu2 = {}'.format(mu1),
                        output_matrix('sigma1 = ', Sigma1),
                        output_matrix('sigma2 = ', Sigma2)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Network Parameters: \n\n" + "\t" +
           "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
                        'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d), 'n_D = {}'.format(n_D),
                        'n_G = {}'.format(n_G)]) +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Additional Information: \n" +
           "" + "\n")
info.close()


################################################################################################
################################################################################################

# plot the contour of the mixture on the first 2 dimensions
X1, X2 = np.meshgrid(np.linspace(-10., 10.), np.linspace(-10., 10.))
XX = np.array([X1.ravel(), X2.ravel()]).T
Y = (p1 * np.exp(-np.diag(np.matmul(np.matmul(XX - mu1[:2], Sigma1_inv[:2, :2]), (XX - mu1[:2]).T)) / 2) +
     p2 * np.exp(-np.diag(np.matmul(np.matmul(XX - mu2[:2], Sigma2_inv[:2, :2]), (XX - mu2[:2]).T)) / 2))
Y = Y.reshape(X1.shape)
CS = plt.contour(X1, X2, Y)
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.title("Contour plot of the target distribution")
plt.savefig(EXP_DIR + "target.png", format="png")
plt.close()


# plot a real sample from the target
label = np.random.choice([0, 1], size=(mb_size,), p=[p1, p2])
sample = (np.random.multivariate_normal(mu1, Sigma1, mb_size) * (1 - label).reshape((mb_size, 1)) +
          np.random.multivariate_normal(mu2, Sigma2, mb_size) * label.reshape((mb_size, 1)))
plt.scatter(sample[:, 0], sample[:, 1], color='b', alpha=0.4, s=10)
plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
plt.title("Sample from the target distribution")
plt.savefig(EXP_DIR + "target_sample.png", format="png")
plt.close()


################################################################################################
################################################################################################

# convert parameters to tf tensor
mu1_tf = tf.convert_to_tensor(mu1, dtype=tf.float32)
mu2_tf = tf.convert_to_tensor(mu2, dtype=tf.float32)
Sigma1_inv_tf = tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32)
Sigma2_inv_tf = tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

initializer = tf.contrib.layers.xavier_initializer()

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [X_dim], initializer=initializer)

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim_g], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim_g], initializer=initializer)
G_W2 = tf.get_variable('g_w2', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [X_dim], initializer=initializer)

theta_G = [G_W1, G_W2, G_b1, G_b2]


# # target log-density
# def log_density(x):
#     def density(x):
#         return tf.exp(-tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf)) / 2)
#     # return -tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf)) / 2
#     # return tf.reduce_logsumexp(-tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf))/2)
#     # return tf.log(density(x))
#     log_den1 = -tf.matmul(tf.matmul(x - mu1_tf, Sigma1_inv_tf), tf.transpose(x - mu1_tf)) / 2
#     log_den2 = -tf.matmul(tf.matmul(x - mu2_tf, Sigma2_inv_tf), tf.transpose(x - mu2_tf)) / 2
#     return tf.reduce_logsumexp(tf.concat([log_p1_tf + log_den1,
#                                           log_p2_tf + log_den2], 0))


# log densities for a collection of samples (G_sample)
def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2 - np.log(Sigma1_det) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                        np.log(p2) + log_den2], 0), 0), 1)
    # return log_den1


# Score function computed from the target distribution
def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, bound=1.):
    np.random.seed(1)
    return np.random.uniform(-bound, bound, size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    out = (tf.matmul(G_h1, G_W2) + G_b2)
    return out


# output dimension of this function is X_dim
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.nn.tanh(tf.matmul(D_h1, D_W2) + D_b2)
    return out


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)
# D_real = discriminator(X)
D_fake = discriminator(G_sample)

# D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
# G_loss = -tf.reduce_mean(D_fake)


Loss = tf.reduce_sum(tf.square(tf.multiply(S_q(G_sample), D_fake) + diag_gradient(D_fake, G_sample)))


# D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
#             .minimize(-D_loss, var_list=theta_D))
# G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
#             .minimize(G_loss, var_list=theta_G))

D_solver = (tf.train.AdamOptimizer(learning_rate=1e-4)
            .minimize(-Loss, var_list=theta_D))
G_solver = (tf.train.AdamOptimizer(learning_rate=1e-4)
            .minimize(Loss, var_list=theta_G))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(tf.gradients(D_fake, G_sample), feed_dict={z: sample_z(mb_size, z_dim)}))
# print(D_W1.eval(session=sess))


D_loss = np.zeros(N)
G_loss = np.zeros(N)
for it in range(N):
    for _ in range(n_D):
        _, D_Loss_curr = sess.run([D_solver, Loss],
                                  feed_dict={z: sample_z(mb_size, z_dim)})
    D_loss[it] = D_Loss_curr

    for _ in range(n_G):
        _, G_Loss_curr = sess.run([G_solver, Loss],
                                  feed_dict={z: sample_z(mb_size, z_dim)})
    G_loss[it] = G_Loss_curr

    if it % 100 == 0:
        samples = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim)})
        print(np.mean(samples, axis=0))
        print("D_loss", it, ":", D_Loss_curr)
        print("G_loss", it, ":", G_Loss_curr)
        plt.scatter(samples[:, 0], samples[:, 1], color='b', alpha=0.4, s=10)
        plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
        plt.title("Samples at iter {0:04d}, with loss {{D: {1:.4f}, G: {2:.4f}}}.".format(it, D_Loss_curr, G_Loss_curr))
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()
        # sqs= sess.run(S_q_s(G_sample), feed_dict={z: sample_z(mb_size, z_dim)})
        # print(it)
        # print(sqs)
        # print(sq1, '\n', sq2, '\n', sq3)
        # print(np.sum(sq1, 0))
        # print(sq_g_samples)
sess.close()


np.savetxt(EXP_DIR + "loss_D.csv", D_loss, delimiter=",")
plt.plot(D_loss)
plt.ylim(ymin=0)
plt.axvline(np.argmin(D_loss), ymax=np.min(D_loss), color="r")
plt.title("loss_D (min at iter {})".format(np.argmin(D_loss)))
plt.savefig(EXP_DIR + "loss_D.png", format="png")
plt.close()

np.savetxt(EXP_DIR + "loss_G.csv", G_loss, delimiter=",")
plt.plot(G_loss)
plt.ylim(ymin=0)
plt.axvline(np.argmin(G_loss), ymax=np.min(G_loss), color="r")
plt.title("loss_G (min at iter {})".format(np.argmin(G_loss)))
plt.savefig(EXP_DIR + "loss_G.png", format="png")
plt.close()


# for it in range(2000):
#     for _ in range(5):
#         X_mb, _ = mnist.train.next_batch(mb_size)
#         _, D_loss_curr, _ = sess.run(
#             [D_solver, D_loss, clip_D],
#             feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
#         )
#     _, G_loss_curr = sess.run(
#         [G_solver, G_loss],
#         feed_dict={z: sample_z(mb_size, z_dim)}
#     )
#     if it % 100 == 0:
#         print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
#               .format(it, D_loss_curr, G_loss_curr))
#         if it % 1000 == 0:
#             samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})
#             fig = plot(samples)
#             plt.savefig('out/{}.png'
#                         .format(str(i).zfill(3)), bbox_inches='tight')
#             i += 1
#             plt.close(fig)
