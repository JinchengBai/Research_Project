# I just simply migrated the code from WassersteinGAN here and commented out some lines we don't need.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 100
# X_dim = 784
X_dim = 3  # dimension of the target distribution, 3 for e.g.
z_dim = 5
h_dim = 20

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
# As a simple example, use a 3-d Gaussian as target distribution
# parameters

mu = np.array([2, 3, 5])
Sigma = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

Sigma_inv = np.linalg.inv(Sigma)
mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)



def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



X = tf.placeholder(tf.float32, shape=[None, X_dim])


initializer = tf.contrib.layers.xavier_initializer()

D_W1 = tf.get_variable('D_w1', [X_dim, h_dim], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim, 1], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [1], initializer=initializer)

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.get_variable('g_w1', [z_dim, h_dim], dtype=tf.float32, initializer=initializer)
G_b1 = tf.get_variable('g_b1', [h_dim], initializer=initializer)
G_W2 = tf.get_variable('g_w2', [h_dim, X_dim], dtype=tf.float32, initializer=initializer)
G_b2 = tf.get_variable('g_b2', [X_dim], initializer=initializer)

theta_G = [G_W1, G_W2, G_b1, G_b2]

# Score function computed from the target distribution
def S_q(x):
    return tf.matmul(mu_tf - x, Sigma_inv_tf)

# target density
def density(x):
    return tf.exp(-tf.matmul(tf.matmul(x - mu_tf, Sigma_inv_tf), tf.transpose(x - mu_tf))/2)


# Score function computed from the target distribution
def S_q(x):
    # return tf.gradients(tf.log(density(x)), x)
    # return tf.map_fn(lambda a: tf.gradients(tf.log(density(a)), a), x)
    return tf.matmul(mu_tf - x, Sigma_inv_tf)


def sample_z(m, n):
    np.random.seed(1)
    return np.random.uniform(-10., 10., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    out = (tf.matmul(G_h1, G_W2) + G_b2)
    return out


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.nn.tanh(tf.matmul(D_h1, D_W2) + D_b2)
    return out


G_sample = generator(z)
# D_real = discriminator(X)
D_fake = discriminator(G_sample)

# D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
# G_loss = -tf.reduce_mean(D_fake)


Loss = tf.reduce_sum(tf.square(S_q(G_sample) * D_fake + tf.gradients(D_fake, G_sample)))


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
#
# sq = sess.run(S_q(tf.expand_dims(mu_tf, 0)))
# print(sq)
# sess.close()
# samples = sess.run(G_sample, feed_dict={z: sample_z(100, z_dim)})
# sq_g_samples = sess.run(S_q(G_sample), feed_dict={z: sample_z(100, z_dim)})
# print(np.mean(samples, axis=0))
# print(sq_g_samples)

'''
if not os.path.exists('out/'):
    os.makedirs('out/')
i = 0
'''


# print(sess.run(tf.gradients(D_fake, G_sample), feed_dict={z: sample_z(1, z_dim)}))
# print(D_W1.eval(session=sess))

for it in range(10000):
    for _ in range(5):
        _, D_Loss_curr = sess.run([D_solver, Loss],
                                  feed_dict={z: sample_z(mb_size, z_dim)})

    for _ in range(5):
        _, G_Loss_curr = sess.run([G_solver, Loss],
                                  feed_dict={z: sample_z(mb_size, z_dim)})

    if it % 100 == 0:
        samples = sess.run(G_sample, feed_dict={z: sample_z(100, z_dim)})
        sq_g_samples = sess.run(S_q(G_sample), feed_dict={z: sample_z(100, z_dim)})
        print(sq_g_samples)
        print(np.mean(samples, axis=0))
        print("D_loss", it, ":", D_Loss_curr)
        print("G_loss", it, ":", G_Loss_curr)

sess.close()


'''
for it in range(2000):
    for _ in range(5):
        X_mb, _ = mnist.train.next_batch(mb_size)
        _, D_loss_curr, _ = sess.run(
            [D_solver, D_loss, clip_D],
            feed_dict={X: X_mb, z: sample_z(mb_size, z_dim)}
        )
    _, G_loss_curr = sess.run(
        [G_solver, G_loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )
    if it % 100 == 0:
        print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss_curr, G_loss_curr))
        if it % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})
            fig = plot(samples)
            plt.savefig('out/{}.png'
                        .format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

  '''

