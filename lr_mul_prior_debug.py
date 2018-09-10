"""

Stein_GAN: lr_mul_prior_debug.py

Created on 9/9/18 9:51 PM

@author: Hanxi Sun

"""


import tensorflow as tf
import numpy as np
# import matplotlib
# import os
# import sys
from datetime import datetime
import numpy.matlib as nm
import scipy.io
from sklearn.model_selection import train_test_split

# ON_SERVER = False

# if ON_SERVER:
#     matplotlib.use('agg')

import matplotlib.pyplot as plt


def now_str():
    now = datetime.now().strftime('%m%d%H%M%S.%f').split('.')
    return "%s%02d" % (now[0], int(now[1]) // 10000)


# if ON_SERVER:
#     # z_dim = int(sys.argv[1])
#     # h_dim_g = h_dim_d = int(sys.argv[2])
#     # ini_var = float(sys.argv[1])
#     # d_blow = int(sys.argv[2])
#     # lr_d_0 = lr_g_0 = float(sys.argv[3])
#     # inter_decay = int(sys.argv[4])
#     # md = int(sys.argv[5])
#     # sd_normal = float(sys.argv[6])
#     # blow_up = int(sys.argv[7])
#     pass
#
# else:
d_blow = 1
blow_up = 2
lr_d_0 = 1e-4
lr_g_0 = 1e-4
inter_decay = 100
ini_var = 2.
ini_var_alpha = .5
# sd_normal = 1


n_iter_prior = 100001
n_iter = 100001


# Target
data = scipy.io.loadmat('data/covertype.mat')
X_input = data['covtype'][:, 1:]
y_input = data['covtype'][:, 0]
y_input[y_input == 2] = -1

N_all = X_input.shape[0]
X_input = np.hstack([X_input, np.ones([N_all, 1])])
d = X_input.shape[1]
D = d + 1

X_dim = D    # dimension of the target distribution
d_dim = d

z_dim = out_dim_g = h_dim_d = h_dim_g = D * blow_up

mb_size_z = 100  # noise batch
mb_size_x = (blow_up - d_blow) * mb_size_z  # G_sample batch
mb_size_t = 100  # batch size for testing

d_prior = D * d_blow
top = (blow_up - d_blow) * D

n_D = 5
n_G = 1
lbd_0 = 100

# create init G_scale
G_scale_init = np.zeros((blow_up, D), dtype=np.float32)
G_scale_init[:, :d_dim] = ini_var
G_scale_init[:, -1] = ini_var_alpha
G_scale_init = G_scale_init.reshape((-1, out_dim_g))

# split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.2, random_state=21)
X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

N = X_train.shape[0]


# DIR = "/scratch/halstead/h/hu478/Stein_GAN" + "/output/temp/" if ON_SERVER else os.getcwd() + "/output/"
ID = now_str()
# EXP = "logit_ini_var={}_".format(ini_var) + "lr={}_".format(lr_d_0)\
#       + "inter_decay={}_".format(inter_decay) + ID
# EXP_DIR = DIR + EXP + "/"
# if not os.path.exists(EXP_DIR):
#     os.makedirs(EXP_DIR)

print("TimeStart: " + ID)

#######################################################################################################################
# tf.reset_default_graph()

initializer = tf.contrib.layers.xavier_initializer()


lbd = tf.placeholder(tf.float32, shape=[])
alpha_power = tf.placeholder(tf.float32, shape=[])
lr_g = tf.placeholder(tf.float32, shape=[])
lr_d = tf.placeholder(tf.float32, shape=[])
lr_id = tf.placeholder(tf.float32, shape=[])


X = tf.placeholder(tf.float32, shape=[None, X_dim])

D_W1 = tf.get_variable('D_w1', [d_prior, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b1 = tf.get_variable('D_b1', [h_dim_d], initializer=initializer)
D_W2 = tf.get_variable('D_w2', [h_dim_d, h_dim_d], dtype=tf.float32, initializer=initializer)
D_b2 = tf.get_variable('D_b2', [h_dim_d], initializer=initializer)
D_W3 = tf.get_variable('D_w3', [h_dim_d, d_prior], dtype=tf.float32, initializer=initializer)
D_b3 = tf.get_variable('D_b3', [d_prior], initializer=initializer)

theta_D_prior = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

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

# G_scale = tf.get_variable('g_scale', [1, out_dim_g], initializer=tf.constant_initializer(ini_var))
G_scale = tf.get_variable('g_scale', initializer=G_scale_init, dtype=tf.float32)
G_location = tf.get_variable('g_location', [1, out_dim_g], initializer=tf.constant_initializer(0.), dtype=tf.float32)

theta_G_all = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_scale, G_location]

Xs = tf.placeholder(tf.float32, shape=[None, d_dim])
ys = tf.placeholder(tf.float32, shape=[None])


def S_q(theta, a0=1, b0=0.01):
    # https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/bayesian_logistic_regression.py

    w = theta[:, :-1]  # (mw, d)
    s = tf.reshape(theta[:, -1], shape=[-1, 1])  # (mw, 1); alpha = s**2

    y_hat = 1. / (1. + tf.exp(- tf.matmul(Xs, tf.transpose(w))))  # (mx, mw); shape(Xs) = (mx, d)
    y = tf.reshape((ys + 1.) / 2., shape=[-1, 1])  # (mx, 1)

    dw_data = tf.matmul(tf.transpose(y - y_hat), Xs)  # (mw, d)
    dw_prior = - s**2 * w / 2.  # (mw, d)
    dw = dw_data * N / mb_size_x + dw_prior  # (mw, d)

    w2 = tf.reshape(tf.reduce_sum(tf.square(w), axis=1), shape=[-1, 1])  # (mw, 1); = wtw
    ds = (2. * a0 - 2 + d) / s - tf.multiply(w2 + 2. * b0, s)  # (mw, 1)

    return tf.concat([dw, ds], axis=1)

    # z0 = sample_z(mb_size_z, z_dim)
    # theta = G_sample
    # w = theta[:, :-1]  # (mw, d)
    # w2 = tf.reshape(tf.reduce_sum(tf.square(w), axis=1), shape=[-1, 1])
    # s = tf.reshape(theta[:, -1], shape=[-1, 1])
    # aa = tf.multiply(w2, s)  # tf.multiply(s, w)
    # out = sess.run([G_sample, w2, s, res], feed_dict={z: z0})
    # ww = out[1]
    # ss = out[2]
    # rr = out[3]
    #
    # print(ww.shape, ss.shape, rr.shape)
    # rr[0, 1]
    # ww[0, 1] - ss[0, 0]

    # w = theta[:, :-1]  # logistic weights, take out the last dimension
    # alpha = tf.exp(theta[:, -1])
    # alpha = tf.Print(alpha, [tf.reduce_max(alpha), tf.reduce_min(alpha)], message="\tS_q-alpha = ")
    #
    # wt = tf.multiply((alpha / 2), tf.reduce_sum(tf.square(w), axis=1))
    #
    # coff = tf.matmul(Xs, tf.transpose(w))
    # y_hat = 1.0 / (1.0 + tf.exp(-coff))
    #
    # dw_data = tf.matmul(tf.transpose((tf.tile(tf.reshape(ys, shape=[-1, 1]), [1, mb_size_x]) + 1.) / 2.0
    #                                  - y_hat), Xs)  # Y \in {-1,1}
    # dw_prior = -tf.multiply(tf.tile(tf.reshape(alpha, [-1, 1]), [1, d_dim]), w)
    #
    # dw = dw_data * 1.0 * N / mb_size_x + dw_prior
    # dw = tf.Print(dw, [tf.reduce_max(dw), tf.reduce_min(dw)], message="\tS_q-dw = ")
    #
    # dalpha = d / 2.0 - wt + (a0 - 1) - b0 * alpha + 1  # the last term is the jacobian term
    # dalpha = tf.Print(dalpha, [tf.reduce_max(dalpha), tf.reduce_min(dalpha)], message="\tS_q-dalpha = ")
    #
    # return tf.concat([dw, tf.reshape(dalpha, [-1, 1])], axis=1)  # % first order derivative

# def S_n(xs):
#     return -xs/sd_normal


def S_prior(xs):
    w = xs[:, :-1]
    a = tf.reshape(tf.abs(xs[:, -1]), shape=[-1, 1])
    da = a*0 - 100
    return tf.concat([tf.multiply(w, a), da], axis=1)


def sample_z(m, n, sd=10., seed=1):
    # np.random.seed(seed)
    # s1 = np.random.normal(0, sd, size=[m, n])
    s1 = np.random.uniform(-sd, sd, size=[m, n])
    return s1


def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def generator(z):
    G_h1 = lrelu(tf.matmul(z, G_W1) + G_b1, 0.1)
    G_h1 = tf.nn.dropout(G_h1, keep_prob=0.8)
    G_h1 = tf.Print(G_h1, [tf.reduce_max(G_h1), tf.reduce_min(G_h1)], message="\tGenerator-G_h1 = ")

    G_h2 = lrelu(tf.matmul(G_h1, G_W2) + G_b2, 0.1)
    G_h2 = tf.nn.dropout(G_h2, keep_prob=0.8)
    G_h2 = tf.Print(G_h2, [tf.reduce_max(G_h2), tf.reduce_min(G_h2)], message="\tGenerator-G_h2 = ")

    G_h3 = tf.matmul(G_h2, G_W3) + G_b3
    G_h3 = tf.Print(G_h3, [tf.reduce_max(G_h3), tf.reduce_min(G_h3)], message="\tGenerator-G_h3 = ")

    out = tf.multiply(G_h3, G_scale) + G_location
    out = tf.Print(out, [tf.reduce_max(out), tf.reduce_min(out)], message="\tGenerator-out = ")
    return out


# def prior(m, a0=1, b0=100):
#     alpha0 = tf.random_gamma(shape=[m*blow_up, 1], alpha=a0, beta=b0)
#     theta = tf.random_normal(shape=[m*blow_up, d_dim])
#     theta = tf.multiply(theta, tf.sqrt(1/alpha0))
#     return tf.reshape(tf.concat([theta, alpha0], axis=1), shape=[-1, D*blow_up])


def discriminator_stein(x):
    SD_h1 = tf.nn.relu(tf.matmul(x, SD_W1) + SD_b1)
    SD_h1 = tf.Print(SD_h1, [tf.reduce_max(SD_h1), tf.reduce_min(SD_h1)], message="\tDiscriminator-D_h1 = ")

    SD_h2 = tf.nn.relu(tf.matmul(SD_h1, SD_W2) + SD_b2)
    SD_h2 = tf.Print(SD_h2, [tf.reduce_max(SD_h2), tf.reduce_min(SD_h2)], message="\tDiscriminator-D_h2 = ")

    out = (tf.matmul(SD_h2, SD_W3) + SD_b3)
    out = tf.Print(out, [tf.reduce_max(out), tf.reduce_min(out)], message="\tDiscriminator-out = ")

    return out


def discriminator_prior(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    out = (tf.matmul(D_h2, D_W3) + D_b3)
    return out


def diag_gradient(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(X_dim)], axis=0)
    return tf.transpose(dg)


def diag_gradient_n(y, x):
    dg = tf.stack([tf.gradients(y[:, i], x)[0][:, i] for i in range(d_prior)], axis=0)
    return tf.transpose(dg)


# def evaluation(theta, n_test=len(y_test)):
#     theta = theta[:, :-1]
#     M = mb_size_t
#
#     prob = np.zeros([n_test, M])
#     for t in range(M):
#         coff = np.multiply(y_test, np.sum(-1 * np.multiply(nm.repmat(theta[t, :], n_test, 1), X_test), axis=1))
#         prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))
#
#     prob = np.mean(prob, axis=1)
#     accu = np.mean(prob > 0.5)
#     llh = np.mean(np.log(prob))
#     return [accu, llh]


def evaluation(theta, X_t=X_test, y_t=y_test):
    w = theta[:, :-1]
    y = y_t.reshape([-1, 1])
    coff = - np.matmul(y * X_t, w.T)

    prob = np.mean(1. / (1 + np.exp(coff)), axis=1)
    acc = np.mean(prob > .5)
    llh = np.mean(np.log(prob))
    return acc, llh


G_sample = tf.reshape(generator(z)[:, :top], shape=[-1, X_dim])
G_prior = tf.reshape(generator(z)[:, top:], shape=[-1, X_dim])

D_fake_stein = discriminator_stein(G_sample)
D_fake_prior = discriminator_prior(G_prior)

loss1 = tf.expand_dims(tf.reduce_sum(tf.multiply(S_q(G_sample), D_fake_stein), 1), 1)
loss1_n = tf.expand_dims(tf.reduce_sum(tf.multiply(S_prior(G_prior), D_fake_prior), 1), 1)

loss2 = tf.expand_dims(tf.reduce_sum(diag_gradient(D_fake_stein, G_sample), axis=1), 1)
loss2_n = tf.expand_dims(tf.reduce_sum(diag_gradient_n(D_fake_prior, G_prior), axis=1), 1)

# Loss = tf.abs(tf.reduce_mean(loss1 + loss2)) - lbd * tf.reduce_mean(tf.square(D_fake_stein))
Loss_alpha = tf.abs(tf.reduce_mean(alpha_power * loss1 + loss2)) - (lbd * tf.reduce_mean(tf.square(D_fake_stein)))
# Loss_grad_D = tf.gradients(Loss_alpha, SD_W1)
# Loss_grad_G = tf.gradients(Loss_alpha, G_W1)
Sq_grad_D = tf.gradients(S_q(G_sample), SD_W1)
Sq_grad_G = tf.gradients(S_q(G_sample), G_W1)
Loss_alpha = tf.Print(Loss_alpha, [Loss_alpha,
                                   # tf.reduce_max(Sq_grad_D), tf.reduce_min(Sq_grad_D),
                                   tf.reduce_max(Sq_grad_G), tf.reduce_min(Sq_grad_G)],
                      message="Loss_alpha & gradients: ")

Loss_n = tf.abs(tf.reduce_mean(loss1_n + loss2_n)) - lbd * tf.reduce_mean(tf.square(D_fake_prior))

# with alpha power to the density
D_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_alpha, var_list=theta_D_stein)
D_solver_n = tf.train.GradientDescentOptimizer(learning_rate=lr_d).minimize(-Loss_n, var_list=theta_D_prior)

# D_solver = tf.train.RMSPropOptimizer(learning_rate=lr_d).minimize(-Loss_n - Loss_alpha, var_list=theta_D)

G_solver_a = tf.train.GradientDescentOptimizer(learning_rate=lr_g).minimize(Loss_alpha, var_list=theta_G_all)
# loss_id = tf.reduce_mean(tf.square(z - generator(z)))
# G_solver_id = tf.train.GradientDescentOptimizer(learning_rate=lr_id).minimize(loss_id, var_list=theta_G_all)


saver = tf.train.Saver(var_list=theta_G_all)


#######################################################################################################################


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Global initialization done!")

# check initial alpha range
init_sample, _ = sess.run([G_sample, G_scale], feed_dict={z: sample_z(10000, out_dim_g)})
init_alpha = init_sample[:, -1]
print("Initial values: ", np.mean(init_alpha), np.std(init_alpha), np.max(init_alpha), np.min(init_alpha))

alpha_1 = 0.01
lbd_1 = lbd_0
lr_g_1 = lr_g_0
lr_d_1 = lr_d_0


G_loss = np.zeros(n_iter)
D_loss = np.zeros(n_iter)
loglik = np.zeros(n_iter // 10)
acc = np.zeros(n_iter // 10)

G_Loss_curr = D_Loss_curr = None


# saver.restore(sess, DIR + "generator_ini.ckpt")
# for it in range(n_iter_prior):
#     _, loss_id_curr = sess.run([G_solver_id, loss_id], feed_dict={z: sample_z(10000, out_dim_g, sd=200), lr_id: 1e-4})
#     if it % 1000 == 0:
#         print(loss_id_curr)
#
# print("Initialization of Generator done!")


for it in range(n_iter):
    print(it)

    batch = [i % N for i in range(it * mb_size_x, (it + 1) * mb_size_x)]

    X_b = X_train[batch, :]
    y_b = y_train[batch]

    for i in range(n_D):
        print('\tDiscriminator', i)
        _, _, D_Loss_curr = sess.run([D_solver_a, D_solver_n, Loss_alpha],
                                     feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size_z, z_dim),
                                                lbd: lbd_0, alpha_power: alpha_1, lr_d: lr_d_1})
        if np.isnan(D_Loss_curr):
            print("NAN D_loss:", it, "-", i)
            break

    if np.isnan(D_Loss_curr):
        print("D_loss:", it)
        break

    D_loss[it] = D_Loss_curr

    # sess.run(G_sample,
    #          feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size_z, z_dim),
    #                     lbd: lbd_0, alpha_power: alpha_1, lr_d: lr_d_1})

    # max_gradient, max_sq = sess.run(
    #     [tf.reduce_max(tf.abs(tf.gradients(Loss_alpha, G_W1))), tf.reduce_max(S_q(G_sample))],
    #     feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size_z, z_dim), lbd: lbd_0, alpha_power: alpha_1, lr_d: lr_d_1})
    #
    # print(it, "pre max_gradient", max_gradient)
    # print(it, "pre max_sq", max_sq)

    # train Generator

    print('\tGenerator')
    _, G_Loss_curr = sess.run([G_solver_a, Loss_alpha],
                              feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size_z, z_dim),
                                         lbd: lbd_0, alpha_power: alpha_1, lr_g: lr_g_1})

    if np.isnan(G_Loss_curr):
        print("NAN G_loss:", it)
        break

    G_loss[it] = G_Loss_curr

    print("\t", it, " losses:", D_Loss_curr, G_Loss_curr)

    # max_gradient, max_sq = sess.run(
    #     [tf.reduce_max(tf.abs(tf.gradients(Loss_alpha, G_W1))), tf.reduce_max(S_q(G_sample))],
    #     feed_dict={Xs: X_b, ys: y_b, z: sample_z(mb_size_z, z_dim), lbd: lbd_0, alpha_power: alpha_1, lr_d: lr_d_1})
    #
    # print(it, "post max_gradient", max_gradient)
    # print(it, "post max_sq", max_sq)

    if it % inter_decay == 0:
        alpha_1 = np.min((alpha_1 + 0.01, 1))  # set alpha_1 = 1 would be original density
        # print("\tDecay")

    if it % 10 == 0:
        # print("\tEvaluation")
        post, loss_curr = sess.run([G_sample, Loss_alpha],
                                   feed_dict={Xs: X_b, ys: y_b,
                                              z: sample_z(mb_size_z, z_dim), lbd: lbd_0, alpha_power: 1})
        post_eval = evaluation(post)
        acc[it//10] = post_eval[0]
        loglik[it//10] = post_eval[1]
        print('\t', it, "loss:", loss_curr, ":Accuracy:", post_eval[0], "\n" +
              "\t\tLoglik:", post_eval[1], "\n",
              "\t\tmean:", np.mean(post), "std:", np.std(post))
        # print(it, "loss:", loss_curr, ":Accuracy:", post_eval[0], "\n" +
        #       "\tLoglik:", post_eval[1], "\n",
        #       "\tmean:", np.mean(post), "std:", np.std(post))


# np.savetxt(EXP_DIR + "_loss_D.csv", D_loss, delimiter=",")
# plt.plot(D_loss)
# plt.axvline(np.argmin(D_loss), ymax=np.min(D_loss), color="r")
# plt.title("loss_D (min at iter {})".format(np.argmin(D_loss)))
# plt.savefig(EXP_DIR + "_loss_D.png", format="png")
# plt.close()
#
# np.savetxt(EXP_DIR + "_loss_G.csv", G_loss, delimiter=",")
# plt.plot(G_loss)
# plt.axvline(np.argmin(G_loss), ymax=np.min(G_loss), color="r")
# plt.title("loss_G (min at iter {})".format(np.argmin(G_loss)))
# plt.savefig(EXP_DIR + "_loss_G.png", format="png")
# plt.close()


print("TimeEnds: " + now_str())

# saver = tf.train.Saver()
# saver.save(sess, EXP_DIR + "model.ckpt")


