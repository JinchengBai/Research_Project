import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os

DIR = os.getcwd() + "/output/"
EXP = "080518-CompareSVGD-1"
EXP_DIR = DIR + EXP + "/"
if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

mb_size = 100
z_dim = 2  # dimension of the target distribution
lr = 0.1
num_iter = 5000


# parameters
p1, p2 = 0.5, 0.5
mu1, mu2 = np.array([3, 3]), np.array([-3, -3])
Sigma1 = Sigma2 = np.matrix([[1, 0], [0, 1]])
Sigma1_inv = np.linalg.inv(Sigma1)
Sigma2_inv = np.linalg.inv(Sigma2)
Sigma1_det = np.linalg.det(Sigma1)
Sigma2_det = np.linalg.det(Sigma2)


################################################################################################
################################################################################################


# def output_matrix(prefix, matrix):
#     return prefix + matrix.__str__().replace('\n', '\n\t'+' '*len(prefix))
#
#
# info = open(EXP_DIR + "info.txt", 'w')
# info.write("Compare SVGD" +
#            '\n\n' + ("=" * 80 + '\n') + '\n' +
#            "Description: " +
#            '\n\n' + ("=" * 80 + '\n') + '\n' +
#            "Model Parameters: \n\n" + "\t" +
#            "\n\t".join(['p1 = {}'.format(p1), 'p2 = {}'.format(p2),
#                         'mu1 = {}'.format(mu1), 'mu2 = {}'.format(mu1),
#                         output_matrix('sigma1 = ', Sigma1),
#                         output_matrix('sigma2 = ', Sigma2)]) +
#            '\n\n' + ("=" * 80 + '\n') + '\n' +
#            "Network Parameters: \n\n" + "\t" +
#            "\n\t".join(['mb_size = {}'.format(mb_size), 'X_dim = {}'.format(X_dim), 'z_dim = {}'.format(z_dim),
#                         'h_dim_g = {}'.format(h_dim_g), 'h_dim_d = {}'.format(h_dim_d), 'n_D = {}'.format(n_D),
#                         'n_G = {}'.format(n_G)]) +
#            '\n\n' + ("=" * 80 + '\n') + '\n' +
#            "Additional Information: \n" +
#            "" + "\n")
# info.close()


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


################################################################################################
# show samples from target
label = np.random.choice([0, 1], size=(mb_size,), p=[p1, p2])
true_sample = (np.random.multivariate_normal(mu1, Sigma1, mb_size) * (1 - label).reshape((mb_size, 1)) +
          np.random.multivariate_normal(mu2, Sigma2, mb_size) * label.reshape((mb_size, 1)))
for i in xrange(z_dim):
    plt.scatter(true_sample[:, i], np.zeros(mb_size), color='b', alpha=0.2, s=10)
    plt.axvline(x=mu1[i])
    plt.axvline(x=mu2[i])
    j = i + 1
    plt.title("Dim {} of true sample from the target distribution". format(j))
    plt.savefig(EXP_DIR + "_target_sample_{}.png". format(j), format="png")
    plt.close()


################################################################################################

# convert parameters to tf tensor
mu1_tf = tf.convert_to_tensor(mu1, dtype=tf.float32)
mu2_tf = tf.convert_to_tensor(mu2, dtype=tf.float32)
Sigma1_inv_tf = tf.convert_to_tensor(Sigma1_inv, dtype=tf.float32)
Sigma2_inv_tf = tf.convert_to_tensor(Sigma2_inv, dtype=tf.float32)


z = tf.placeholder(tf.float32, shape=[None, z_dim])


def svgd_kernel(x):  # adopted from reference
    XY = tf.matmul(x, tf.transpose(x))
    X2_ = tf.reduce_sum(tf.square(x), axis=1)
    x2 = tf.reshape(X2_, shape=(tf.shape(x)[0], 1))
    X2e = tf.tile(x2, [1, tf.shape(x)[0]])
    H = tf.subtract(tf.add(X2e, tf.transpose(X2e)), 2 * XY)

    V = tf.reshape(H, [-1, 1])

    # median distance
    def get_median(v):
        v = tf.reshape(v, [-1])
        m = tf.shape(v)[0] // 2
        return tf.nn.top_k(v, m).values[m - 1]

    h = get_median(V)
    h = tf.sqrt(0.5 * h / tf.log(tf.cast(tf.shape(x)[0], tf.float32) + 1.0))

    # compute the rbf kernel
    Kxy = tf.exp(-H / h ** 2 / 2.0)

    dxkxy = -tf.matmul(Kxy, x)
    sumkxy = tf.expand_dims(tf.reduce_sum(Kxy, axis=1), 1)
    dxkxy = tf.add(dxkxy, tf.multiply(x, sumkxy)) / (h ** 2)

    return Kxy, dxkxy


def log_densities(xs):
    log_den1 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu1_tf, Sigma1_inv_tf),
                                        tf.transpose(xs - mu1_tf))) / 2
    log_den2 = - tf.diag_part(tf.matmul(tf.matmul(xs - mu2_tf, Sigma2_inv_tf),
                                        tf.transpose(xs - mu2_tf))) / 2
    return tf.expand_dims(tf.reduce_logsumexp(tf.stack([np.log(p1) + log_den1,
                                                        np.log(p2) + log_den2], 0), 0), 1)


def S_q(xs):
    return tf.gradients(log_densities(xs), xs)[0]


def sample_z(m, n, bound=5.):
    return np.random.uniform(-bound, bound, size=[m, n])


log_p_grad = S_q(z)
out = svgd_kernel(z)
kernel_matrix, kernel_gradients = out[0], out[1]
grad_theta = (tf.matmul(kernel_matrix, log_p_grad) + kernel_gradients)/mb_size
z_np = sample_z(mb_size, z_dim)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
for it in xrange(num_iter):
    grad_theta_ = sess.run(grad_theta, feed_dict={z: z_np})
    z_np = z_np + lr * grad_theta_

    if it % 100 == 0:
        # z_test = np.reshape(z_np, (mb_size, z_dim))
        sample_mean = np.mean(z_np, axis=0)
        sample_sd = np.std(z_np, axis=0)
        print(it, ":", sample_mean, sample_sd)

        plt.subplot(322)
        # plt.title("Samples of dim 2")
        plt.scatter(true_sample[:, 1], np.ones(mb_size), color='purple', alpha=0.2, s=10)
        plt.scatter(z_np[:, 1], np.zeros(mb_size), color='b', alpha=0.2, s=10)
        # plt.plot(samples[:, 0], np.zeros(100), 'ro', color='b', ms=1)
        plt.axvline(mu1[1], color='r')
        plt.axvline(mu2[1], color='r')

        plt.subplot(323)
        # plt.title("Histogram of dim 1")
        num_bins = 50
        # the histogram of the data
        _, bins, _ = plt.hist(z_np[:, 0], num_bins, normed=1, facecolor='green', alpha=0.5)
        # add a 'best fit' line
        y = p1 * mlab.normpdf(bins, mu1[0], Sigma1[0, 0]) + p2 * mlab.normpdf(bins, mu2[0], Sigma2[0, 0])
        plt.plot(bins, y, 'r--')
        plt.ylabel('Probability')

        plt.subplot(324)
        # plt.title("Histogram of dim 2")
        num_bins = 50
        # the histogram of the data
        _, bins, _ = plt.hist(z_np[:, 1], num_bins, normed=1, facecolor='green', alpha=0.5)
        # add a 'best fit' line
        y = p1 * mlab.normpdf(bins, mu1[1], Sigma1[1, 1]) + p2 * mlab.normpdf(bins, mu2[1], Sigma2[1, 1])
        plt.plot(bins, y, 'r--')
        plt.ylabel('Probability')

        plt.subplot(325)
        # plt.title("Scatter plot")
        plt.scatter(z_np[:, 0], z_np[:, 1], color='b', alpha=0.4, s=10)
        plt.scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color="r")

        plt.subplot(321)
        # plt.title("Samples of dim 1")
        plt.scatter(true_sample[:, 0], np.ones(mb_size), color='purple', alpha=0.2, s=10)
        plt.scatter(z_np[:, 0], np.zeros(mb_size), color='b', alpha=0.2, s=10)
        # plt.plot(samples[:, 0], np.zeros(100), 'ro', color='b', ms=1)
        plt.axvline(mu1[0], color='r')
        plt.axvline(mu2[0], color='r')
        plt.title(
            "iter {}, mean: {}, sd: {}".format(it, np.around(sample_mean, 4), np.around(sample_sd, 4)))
        plt.savefig(EXP_DIR + "iter {}".format(it))
        plt.close()
sess.close()
