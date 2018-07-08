# I just simply migrated the code from WassersteinGAN here and commented out some lines we don't need.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
# X_dim = 784
X_dim = 3  # dimension of the target distribution, 3 for e.g.
z_dim = 10
h_dim = 128

# mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
# As a simple example, use a 3-d Gaussian as target distribution
# parameters
mu = np.array([0,0,0])
Sigma = np.matrix([[1,-1,0],[-1,4,0.5],[0,0.5,2]])
Sigma_inv = np.linalg.inv(Sigma)
mu_tf = tf.convert_to_tensor(mu, dtype=tf.float32)
Sigma_inv_tf = tf.convert_to_tensor(Sigma_inv, dtype=tf.float32)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


z = tf.placeholder(tf.float32, shape=[None, z_dim])

G_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

# Score function computed from the target distribution
def S_q(x):
    return tf.matmul(mu_tf - x, Sigma_inv_tf)

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
#    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    out = tf.matmul(G_h1, G_W2) + G_b2
#    G_prob = tf.nn.sigmoid(G_log_prob)
#    return G_prob
    return out


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out


G_sample = generator(z)
# D_real = discriminator(X)
D_fake = discriminator(G_sample)

# D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
# G_loss = -tf.reduce_mean(D_fake)

Loss = tf.reduce_mean(S_q(G_sample) * D_fake + tf.gradients(D_fake, [G_sample]))
Loss *= Loss

# D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
#             .minimize(-D_loss, var_list=theta_D))
# G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
#             .minimize(G_loss, var_list=theta_G))
D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-Loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(Loss, var_list=theta_G))



clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(5000):
    _, Loss_curr, _ = sess.run([D_solver, Loss, Clip_D],
    feed_dict={z: sample_z(mb_size, z_dim)}
    )
    _, Loss_curr = sess.run(
        [G_solver, Loss],
        feed_dict={z: sample_z(mb_size, z_dim)}
    )
    if it % 100 == 0:
        samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})
        samples = np.array(samples)
        print np.mean(samples, axis=0)



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
