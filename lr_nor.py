"""

Stein_GAN: Double_cross.py

Created on 8/23/18 10:05 PM

@author: Tianyang Hu

"""

ON_SERVER = False
# ON_SERVER = True

import tensorflow as tf
import numpy as np
import matplotlib
import os
import sys
from datetime import datetime
import numpy.matlib as nm
import scipy.io
from sklearn.model_selection import train_test_split

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
    md = int(sys.argv[5])
    sd_normal = float(sys.argv[6])
    blow_up = int(sys.argv[7])

else:
    # d_normal = 10
    lr_d_0 = lr_g_0 = 1e-5
    inter_decay = 100
    ini_var = 1.
    sd_normal = 1
    blow_up = 4


mb_size_w = 100  # coefficient batch
mb_size_x = mb_size_w * blow_up  # G_sample batch
mb_size_t = 100
n_iter_prior = 10001
n_iter = 100001


# Target
data = scipy.io.loadmat('data/covertype.mat')
X_input = data['covtype'][:, 1:]
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = -1

N = X_input.shape[0]
X_input = np.hstack([X_input, np.ones([N, 1])])
d = X_input.shape[1]
D = d + 1

X_dim = D    # dimension of the target distribution
d_dim = d

z_dim = out_dim_g = h_dim_d = h_dim_g = D * blow_up
n_D = 5
n_G = 1
lbd_0 = 10
# top = out_dim_g - d_normal

# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=42)
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)


DIR = "/scratch/halstead/h/hu478/Stein_GAN" + "/output/temp/" if ON_SERVER else os.getcwd() + "/output/"
ID = now_str()
EXP = "logit_ini_var={}_".format(ini_var) + "lr={}_".format(lr_d_0)\
      + "inter_decay={}_".format(inter_decay) + "sd_normal={}_".format(sd_normal) + ID
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

print("TimeStart: " + ID)

PLT_SHOW = False

#######################################################################################################################
initializer = tf.contrib.layers.xavier_initializer()

lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])
lr_g = tf.placeholder(tf.float32, shape=[])
lr_d = tf.placeholder(tf.float32, shape=[])


X = tf.placeholder(tf.float32, shape=[None, X_dim])

# D_W1 = tf.get_variable('D_w1', [d_normal, h_dim_d], dtype=tf.float32, initializer=initializer)
# D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
# D_W2 = tf.get_variable('D_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
# D_b2 = tf.get_variable('D_b2', [h_dim_d], initializer=initializer)
# D_W3 = tf.get_variable('D_w3', [h_dim_d, d_normal], dtype=tf.float32, initializer=initializer)
# D_b3 = tf.get_variable('D_b3', [d_normal], initializer=initializer)
#
# theta_D_normal = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

SD_W1 = tf.get_variable('SD_w1', [X_dim, h_dim_d], dtype=tf.float32, initializer=initializer)
SD_b1 = tf.get_variable('SD_b1', [h_dim_d], initializer=initializer)
SD_W2 = tf.get_variable('SD_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
SD_b2 = tf.get_variable('SD_b2', [h_dim_d], initializer=initializer)
SD_W3 = tf.get_variable('SD_w3', [h_dim_d, X_dim], dtype=tf.float32, initializer=initializer)
SD_b3 = tf.get_variable('SD_b3', [X_dim], initializer=initializer)

theta_D_stein = [SD_W1, SD_W2, SD_W3, SD_b1, SD_b2, SD_b3]
# theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, SD_W1, SD_W2, SD_W3, SD_b1, SD_b2, SD_b3]


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

Xs = tf.placeholder(tf.float32, shape=[None, d_dim])
ys = tf.placeholder(tf.float32, shape=[None])


def S_q(theta, a0=1, b0=0.01):
    # https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/bayesian_logistic_regression.py

    w = theta[:, :-1]  # logistic weights, take out the last dimension
    alpha = tf.exp(theta[:, -1])  # the last column is logalpha
    d = d_dim

    wt = tf.multiply((alpha / 2), tf.reduce_sum(tf.square(w), axis=1))

    coff = tf.matmul(Xs, tf.transpose(w))
    y_hat = 1.0 / (1.0 + tf.exp(-coff))

    dw_data = tf.matmul(tf.transpose((tf.tile(tf.reshape(ys, shape=[-1, 1]),
                                              [1, mb_size_w*blow_up]) + 1) / 2.0 - y_hat), Xs)
    # Y \in {-1,1}
    dw_prior = -tf.multiply(tf.tile(tf.reshape(alpha, [-1, 1]), [1, d_dim]), w)
    dw = dw_data * 1.0 * N / mb_size_x + dw_prior  # re-scale

    dalpha = d / 2.0 - wt + (a0 - 1) - b0 * alpha + 1  # the last term is the jacobian term

    return tf.concat([dw, tf.reshape(dalpha, [-1, 1])], axis=1)  # % first order derivative


def S_n(xs):
    return -xs/sd_normal


def sample_z(m, n, sd=10.):
    s1 = np.random.normal(0, sd, size=[m, n])
    # s1 = np.random.uniform(-sd, sd, size=[m, n])
    return s1


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def generator(z):
    G_h1 = lrelu(tf.matmul(z, G_W1) + G_b1, 0.1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h2 = lrelu(tf.matmul(G_h1, G_W2) + G_b2, 0.1)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    out = tf.multiply(G_h3, G_scale) + G_location
    return out


def prior(m, a0=1, b0=100):
    alpha0 = tf.random_gamma(shape=[m*blow_up, 1], alpha=a0, beta=b0)
    theta = tf.random_normal(shape=[m*blow_up, d_dim])
    theta = tf.multiply(theta, tf.sqrt(1/alpha0))
    return tf.reshape(tf.concat([theta, alpha0], axis=1), shape=[-1, D*blow_up])


def discriminator_stein(x):
    SD_h1 = tf.nn.relu(tf.matmul(x, SD_W1) + SD_b1)
    SD_h2 = tf.nn.relu(tf.matmul(SD_h1, SD_W2) + SD_b2)
    # D_h1 = tf.Print(D_h1, [D_h1], message="Discriminator-"+"D_h1"+"-values:")
    out = (tf.matmul(SD_h2, SD_W3) + SD_b3)
    # out = tf.Print(out, [out], message="Discriminator-"+"out"+"-values:")
    return out


# def discriminator_normal(x):
#     D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
#     D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
#     out = (tf.matmul(D_h2, D_W3) + D_b3)
#     return out


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


def diag_gradient_n(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(d_normal)], axis=0)
    return tf.transpose(dg)


# def evaluation(theta, n_test=len(y_test)):
#     theta = theta[:, :-1]
#     M = mb_size_t
#
#     prob = tf.zeros([n_test, M])
#     for t in range(M):
#         coff = tf.multiply(y_test_tf,
#                            tf.reduce_sum(-1 * tf.multiply(
#                                tf.tile(tf.expand_dims(theta[t, :], 0), [n_test, 1]), X_test_tf), axis=1))
#         prob[:, t] = (tf.ones(n_test) / (1 + tf.exp(coff)))
#
#     prob = tf.reduce_mean(prob, axis=1)
#     acc = tf.reduce_mean(prob > 0.5)
#     llh = tf.reduce_mean(np.log(prob))
#     return [acc, llh]

def evaluation(theta, n_test=len(y_test)):
    theta = theta[:, :-1]
    M = mb_size_t

    prob = np.zeros([n_test, M])
    for t in range(M):
        coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
        prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))

    prob = np.mean(prob, axis=1)
    acc = np.mean(prob > 0.5)
    llh = np.mean(np.log(prob))
    return [acc, llh]


G_sample = tf.reshape(generator(prior(m=mb_size_w)), shape=[-1, X_dim])

# alpha0 = tf.zeros(shape=[5*4, 1])
# theta = tf.ones(shape=[5*4, 2])
# res = tf.reshape(tf.concat([theta, alpha0], axis=1), shape=[-1, (1+2)*4])
# G = tf.reshape(res, shape=[-1, (1+2)])
# sess = tf.Session()
# out = sess.run([res, G])
# out[0]
# out[1]

# G_normal = generator(z)[:, top:]
D_fake_stein = discriminator_stein(G_sample)
# D_fake_normal = discriminator_normal(G_normal)


loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake_stein), 1), 1)
# loss1_n = tf.expand_dims(tf.reduce_sum(tf.multiply(S_n(G_normal), D_fake_normal), 1), 1)

loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake_stein, G_sample), axis=1), 1)
# loss2_n = tf.expand_dims(tf.reduce_sum(diag_gradient_n(D_fake_normal, G_normal), axis=1), 1)

Loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))
Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))

# Loss_n = tf.abs(tf.reduce_mean(loss1_n + loss2_n)) - lbd * tf.reduce_mean(tf.square(D_fake_normal))


# with alpha power to the density
D_solver_a = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D_stein)

# D_solver_n = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_n, var_list=theta_D_normal)

# D_solver = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_n - Loss_alpha, var_list=theta_D)

G_solver_a = tf.train.RMSPropOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G_all)


G_solver_id = tf.train.RMSPropOptimizer(learning_rate=lr_g).minimize(
    tf.reduce_sum(tf.square(z - generator(z))), var_list=theta_G_all)
#######################################################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")


alpha_1 = 0.01
lbd_1 = lbd_0
lr_g_1 = lr_g_0
lr_d_1 = lr_d_0


G_loss = np.zeros(n_iter)
D_loss = np.zeros(n_iter)

G_Loss_curr = D_Loss_curr = None


for it in range(n_iter_prior):
    sess.run(G_solver_id, feed_dict={z: sample_z(1000, out_dim_g, sd=200), lr_g: lr_g_1})

print("Initialization of Generator done!")


for it in range(n_iter):

    batch = [i % N for i in range(it * mb_size_x, (it + 1) * mb_size_x)]

    X_b = X_train[batch, :]
    y_b = y_train[batch]

    for _ in range(n_D):
        # _, _, D_Loss_curr = sess.run([D_solver_a, D_solver_n, Loss_alpha],
        #                              feed_dict={z: prior(mb_size_w), lbd: lbd_0,
        #                                         alpha_power: alpha_1, lr_d: lr_d_1})
        _, D_Loss_curr = sess.run([D_solver_a, Loss_alpha],
                                  feed_dict={Xs: X_b, ys: y_b,
                                             lbd: lbd_0, alpha_power: alpha_1, lr_d: lr_d_1})

    D_loss[it] = D_Loss_curr

    if np.isnan(D_Loss_curr):
        print("D_loss:", it)
        break

    # train Generator
    _, G_Loss_curr = sess.run([G_solver_a, Loss_alpha],
                              feed_dict={Xs: X_b, ys: y_b,
                                         lbd: lbd_0, alpha_power: alpha_1, lr_g: lr_g_1})

    G_loss[it] = G_Loss_curr

    if np.isnan(G_Loss_curr):
        print("G_loss:", it)
        break

    if it % inter_decay == 0:
        alpha_1 = np.min((alpha_1 + 0.01, 1))  # set alpha_1 = 1 would be original density

    if it % 1000 == 0:
        post = sess.run(G_sample, feed_dict={z: prior(mb_size_t)})
        post_eval = evaluation(post)
        print(it, ":Accuracy:", post_eval[0], "\n" + "Loglik:", post_eval[1])


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

