import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    expanded_X = tf.expand_dims(X, 1)  # Nx1xD
    expanded_MU = tf.expand_dims(MU, 0)  # 1xKxD
    distances = tf.reduce_sum(tf.square(tf.subtract(expanded_X, expanded_MU)), 2)
    return distances


def log_GaussPDF(X, mu, sigma_square):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    # sigma_square = tf.sqaure(sigma)

    const = - tf.multiply(tf.cast(dim, tf.float32) / 2, tf.log(2 * np.pi * tf.transpose(sigma_square)))
    dist = distanceFunc(X, mu)
    exp = - tf.divide(dist, 2 * tf.transpose(sigma_square))

    return tf.add(const, exp)


def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    numerator_log = tf.add(log_PDF, tf.transpose(log_pi))
    denominator_log = -hlp.reduce_logsumexp(numerator_log, 1, True)
    return tf.add(numerator_log, denominator_log)


def plotLoss(loss, K, is_valid=False):
    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if is_valid:
        plt.title('validation loss with K = {}'.format(K))
    else:
        plt.title('training loss with K = {}'.format(K))


def GMM(K, is_valid=False):
    print('Enter K = {} -----------------------------------------'.format(K))
    # For Validation set
    train_data = data
    train_batch = num_pts
    if is_valid:
        valid_batch = int(num_pts / 3.0)
        train_batch = num_pts - valid_batch
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        train_data = data[rnd_idx[valid_batch:]]

    X = tf.placeholder(tf.float32, [None, dim])
    MU = tf.Variable(tf.truncated_normal([K, dim], dtype=tf.float32))
    phi = tf.Variable(tf.truncated_normal([K, 1], dtype=tf.float32))
    psi = tf.Variable(tf.truncated_normal([K, 1], dtype=tf.float32))  # tf.Variable(tf.ones([K, 1], dtype=tf.float32)/K)
    sigma_sq = tf.exp(phi)  # Kx1
    log_pi = logsoftmax(psi)  # Kx1

    log_gauss = log_GaussPDF(X, MU, sigma_sq)  # NxK
    log_post = log_posterior(log_gauss, log_pi)  # NxK
    assign_predict = tf.argmax(log_post, 1)
    loss = - tf.reduce_sum(reduce_logsumexp(log_gauss + tf.transpose(log_pi), 1), axis=0)

    adam_op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    train_loss = []
    valid_loss = []
    for i in range(500):
        _, MU_value, log_pi_values, variance, loss_value, log_post_value, predictions = sess.run(
            [adam_op, MU, log_pi, sigma_sq, loss, log_post, assign_predict], feed_dict={X: train_data})
        train_loss.append(loss_value)

        if is_valid:
            val_loss = sess.run(loss, feed_dict={X: val_data})
            valid_loss.append(val_loss)

    #   print('final loss on train data :')
    #   print(train_loss[len(train_loss) - 1])
    # print('final loss on valid data :')
    # print(valid_loss[len(valid_loss) - 1])
    #   print(log_post_value)
    #   print(predictions)

    #   plt.figure(figsize=(6, 4))
    #   plt.style.use('default')
    #   # plt.subplot(121)
    #   if is_valid:
    #     plotLoss(valid_loss, K, is_valid=True)
    #   else:
    #     plotLoss(train_loss, K)

    plt.figure(figsize=(6, 4))
    plt.style.use('default')
    plt.scatter(train_data[:, 0], train_data[:, 1], c=predictions, s=1, alpha=0.8)
    plt.plot(MU_value[:, 0], MU_value[:, 1], 'r^', markersize=8)
    plt.title('K = {}'.format(K))


GMM(1)
GMM(2)
GMM(3)
GMM(4)
GMM(5)

# GMM(5)
# GMM(10)
# GMM(15)
# GMM(20)
# GMM(30)

plt.show()


