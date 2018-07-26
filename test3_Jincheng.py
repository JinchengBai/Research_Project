import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

DIR = os.getcwd() + "/output/"
EXP = "072318-7"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 100
X_dim = 1  # dimension of the target distribution, 3 for e.g.
z_dim = 1
h_dim_d = 50
N, n_D, n_G = 20000, 1, 1  # num of iterations


mu1 = np.array([2])
Sigma1 = np.matrix([3])
Sigma1_inv = np.linalg.inv(Sigma1)


# convert parameters to tf tensor
mu1_tf = tf.convert_to_tensor(mu1, dtype=tf.float32)
Sigma1_inv_tf = tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

initializer = tf.contrib.layers.xavier_initializer()

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [X_dim], initializer=initializer)

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])


G_W1 = tf.get_variable('g_w1', [z_dim, X_dim], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [X_dim], initializer=initializer)

theta_G = [G_W1, G_b1]


def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([log_den1], 0), 0), 1)


# Score function computed from the target distribution
def S_q(xs):
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, bound=1.):
    return np.random.normal(0, 1, size=[m, n])


def generator(z):
    out = tf.matmul(z, G_W1) + G_b1
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
D_fake = discriminator(G_sample)

Loss = tf.reduce_mean(tf.square(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake)
                                             + diag_gradient(D_fake, G_sample), 1)))

D_solver = (tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            .minimize(-Loss, var_list=theta_D))
G_solver = (tf.train.GradientDescentOptimizer(learning_rate=1e-3)
            .minimize(Loss, var_list=theta_G))


# clip_G = [p.assign(tf.clip_by_value(p, -1., 1.)) for p in theta_G]


sess = tf.Session()
sess.run(tf.global_variables_initializer())


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
        # samples_test = [[s] for s in np.linspace(1.5, 2.5, 100, dtype=np.float32)]
        samples_test = np.reshape(np.linspace(-5, 5, 1000, dtype=np.float32), newshape=[1000,1])
        # z_test = [[s] for s in np.linspace(-1., 1., 100, dtype=np.float32)]
        # z_test = np.reshape(np.linspace(-1., 1., 100, dtype=np.float32), newshape=[100,1])
        # discriminators = sess.run(discriminator(samples_test), feed_dict={z: sample_z(mb_size, z_dim)})
        discriminators = sess.run(discriminator(X), feed_dict={X: samples_test})
        # generators = sess.run(generator(z), feed_dict={z: z_test})
        sample_mean = np.mean(samples, axis=0)
        print(sess.run([G_W1,G_b1]))
        print(np.mean(samples, axis=0))
        print("D_loss", it, ":", D_Loss_curr)
        print("G_loss", it, ":", G_Loss_curr)
        # plt.scatter(samples[:, 0], samples[:, 1], color='b')
        # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
        plt.plot()
        plt.title("Samples at iter {0:04d}, with loss {{D: {1:.4f}, G: {2:.4f}}}, sample mean is {3}.".format(it, D_Loss_curr, G_Loss_curr,
                                                                                          sample_mean))
        plt.subplot(212)
        plt.ylim([-1., 1.])
        plt.plot(samples_test, discriminators)
        # plt.plot(z_test, generators)
        plt.subplot(211)
        plt.plot(range(mb_size), samples[:, 0], 'ro', color='b')
        plt.axhline(mu1[0], color='r')
        plt.title("Samples at iter {0:04d}, with loss {{D: {1:.4f}, G: {2:.4f}}}, sample mean is {3}.".format(it, D_Loss_curr, G_Loss_curr,
                                                                                          sample_mean))
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()
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

