import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from tensorflow.python import debug as tf_debug
import os

DIR = os.getcwd() + "/output/"
EXP = "072718N-9"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 1000
X_dim = 1  # dimension of the target distribution, 3 for e.g.
z_dim = 1
h_dim_g = 50
h_dim_d = 50
N, n_D, n_G = 1000, 10, 1  # num of iterations


mu1 = 1
sd = 1
# This is the covariance matrix
Sigma1 = sd * sd
Sigma1_inv = 1/Sigma1

################################################################################################
################################################################################################
tf.reset_default_graph()
info = open(EXP_DIR + "_info.txt", 'w')
info.write("Description: " +
           '\n\n' + ("=" * 80 + '\n') * 3 + '\n' +
           "Model Parameters: \n\t" +
           "\n\t".join(['mu1 = {}'.format(mu1), 'sigma1 = {}'.format(Sigma1)]) +
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
################################################################################################
# convert parameters to tf tensor
mu1_tf = tf.reshape(tf.convert_to_tensor(mu1, dtype=tf.float32), shape=[1])
# mu2_tf = tf.convert_to_tensor(mu2, dtype=tf.float32)
Sigma1_inv_tf = tf.reshape(tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32), shape=[1, 1])
# Sigma2_inv_tf = tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32)


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
# G_W2 = tf.get_variable('g_w2', [h_dim_g, X_dim], dtype=tf.float32, initializer=initializer)
# G_b2 = tf.get_variable('g_b2', [X_dim], initializer=initializer)

theta_G = [G_W1, G_b1]


def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2
    # log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
    #                                     tf.transpose(xs - mu2_tf))) / 2 - np.log(Sigma2_det) / 2
    # return tf.expand_dims(tf.reduce_logsumexp(tf.stack([log_den1], 0), 0), 1)
    return log_den1


# Score function computed from the target distribution
def S_q(xs):
    # return tf.matmul(mu_tf - x, Sigma_inv_tf)
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n):
    # return np.random.uniform(-bound, bound, size=[m, n])
    return np.random.normal(0, 1, size=[m, n])


def generator(z):
    G_h1 = (tf.matmul(z, G_W1) + G_b1)
    # G_h1 = tf.Print(G_h1, [G_h1], message="Generator-"+"G_h1"+"-values:")
    # out = (tf.matmul(G_h1, tf.square(G_W2)) + G_b2)
    # out = tf.Print(out, [out], message="Generator-"+"out"+"-values:")
    # out = out + np.random.normal(0, 5, size=[mb_size, X_dim])
    return G_h1


# output dimension of this function is X_dim
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    # D_h1 = tf.Print(D_h1, [D_h1], message="Discriminator-"+"D_h1"+"-values:")
    out = (tf.matmul(D_h1, D_W2) + D_b2)
    # out = tf.Print(out, [out], message="Discriminator-"+"out"+"-values:")
    return out


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


G_sample = generator(z)
# D_real = discriminator(X)

D_fake = discriminator(G_sample)
G_sample_fake_fake =(G_sample - tf.reduce_mean(G_sample))*2 + tf.reduce_mean(G_sample)
D_fake_fake = discriminator(G_sample_fake_fake)

norm_S = tf.sqrt(tf.reduce_mean(tf.square(D_fake_fake)))


D_fake_d = discriminator(G_sample + np.random.normal(0, 1, [mb_size, 1]))
D_fake_g = D_fake + np.random.normal(0, 0.1, [mb_size, 1])


gradients_d = tf.gradients(D_fake, G_sample)[0]
# gradients_d = tf.Print(gradients_d, [gradients_d], message="gradients_d"+"-values:")
slopes_d = tf.sqrt(tf.reduce_sum(tf.square(gradients_d), reduction_indices=[1]))
# slopes_d = tf.Print(slopes_d, [slopes_d], message="slopes_d"+"-values:")
gradient_penalty_d = 10*tf.reduce_mean((slopes_d-1.0)**2)

range_penalty_g = 10*(generator(tf.constant(1, shape=[1, 1], dtype=tf.float32)) -
                      generator(tf.constant(-1, shape=[1, 1], dtype=tf.float32)))
# range_penalty_g = tf.Print(range_penalty_g, [range_penalty_g], message="range_penalty_g"+"-values:")

l2_penalty_d = tf.reduce_sum(tf.square(D_W1)) + tf.reduce_sum(tf.square(D_W2)) + \
               tf.reduce_sum(tf.square(D_b1)) + tf.reduce_sum(tf.square(D_b2))

loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake), 1), 1)
loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake, G_sample), axis=1), 1)

Loss = tf.abs(tf.reduce_mean(loss1 + loss2))/norm_S

Loss_g = tf.reduce_sum(tf.square(tf.multiply(S_q(G_sample), D_fake_g) + tf.gradients(D_fake_g, G_sample)[0]))
Loss_d = tf.reduce_sum(tf.square(tf.multiply(S_q(G_sample), D_fake_d) + tf.gradients(D_fake_d, G_sample)[0]))


D_solver = (tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            .minimize(-Loss, var_list=theta_D))
G_solver = (tf.train.GradientDescentOptimizer(learning_rate=1e-2)
            .minimize(Loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -100, 100)) for p in theta_D]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

D_loss = np.zeros(N)
G_loss = np.zeros(N)
w = np.zeros(N)
b = np.zeros(N)
D_Loss_curr = G_Loss_curr = w_curr = b_curr = None

for it in range(N):
    for _ in range(n_D):
        _, D_Loss_curr, w_curr, b_curr = sess.run([D_solver, Loss, G_W1, G_b1],
                                                  feed_dict={z: sample_z(mb_size, z_dim)})
    D_loss[it] = D_Loss_curr
    w[it] = w_curr
    b[it] = b_curr

    if np.isnan(D_Loss_curr):
        print("G_loss:", it)
        break

    for _ in range(n_G):
        _, G_Loss_curr = sess.run([G_solver, Loss],
                                  feed_dict={z: sample_z(mb_size, z_dim)})
    G_loss[it] = G_Loss_curr

    if np.isnan(G_Loss_curr):
        print("D_loss:", it)
        break

    if it % 10 == 0:
        noise = sample_z(100, 1)
        x_range = np.reshape(np.linspace(-5, 5, 500, dtype=np.float32), newshape=[500, 1])
        z_range = np.reshape(np.linspace(-5, 5, 500, dtype=np.float32), newshape=[500, 1])
        samples = sess.run(generator(noise.astype(np.float32)))
        disc_func = sess.run(discriminator(x_range))
        gen_func = sess.run(generator(z_range))
        sample_mean = np.mean(samples)
        sample_sd = np.std(samples)
        print(it, ":", sample_mean, sample_sd)
        print("D_loss:", D_Loss_curr)
        print("G_loss:", G_Loss_curr)
        print("w:", G_W1.eval(session=sess), "b:", G_b1.eval(session=sess))
        # plt.scatter(samples[:, 0], samples[:, 1], color='b')
        # plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")
        plt.plot()
        # plt.subplot(212)
        # plt.plot(x_range, disc_func)
        plt.subplot(212)
        plt.ylim(-4, 4)
        plt.plot(x_range, disc_func)
        plt.subplot(211)
        plt.ylim(-3, 5)
        plt.plot(range(100), samples[:, 0], 'ro', color='b', ms=1)
        plt.axhline(mu1, color='r')
        plt.title(
            "iter {0:04d}, {{G: {1:.4f}, mu: {2:.4f}, sd: {3:.4f}}}".format(it, G_Loss_curr, sample_mean, sample_sd))
        plt.savefig(EXP_DIR + "iter {0:04d}".format(it))
        plt.close()

sess.close()


np.savetxt(EXP_DIR + "_loss_D.csv", D_loss, delimiter=",")
plt.plot(D_loss)
plt.ylim(ymin=0)
plt.axvline(np.argmin(D_loss), ymax=np.min(D_loss), color="r")
plt.title("loss_D (min at iter {})".format(np.argmin(D_loss)))
plt.savefig(EXP_DIR + "_loss_D.png", format="png")
plt.close()

np.savetxt(EXP_DIR + "_w.csv", w, delimiter=",")
plt.plot(w)
plt.ylim(-Sigma1-1, Sigma1 + 1)
plt.axhline(y=Sigma1, color="r")
plt.axhline(y=-Sigma1, color="r")
plt.title("Weight")
plt.savefig(EXP_DIR + "_W.png", format="png")
plt.close()

np.savetxt(EXP_DIR + "_b.csv", b, delimiter=",")
plt.plot(b)
plt.ylim(mu1-1, mu1 + 1)
plt.axhline(y=mu1, color="r")
plt.title("Bias")
plt.savefig(EXP_DIR + "_B.png", format="png")
plt.close()
