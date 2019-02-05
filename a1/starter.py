import tensorflow as tf
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        print("in load data")
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def MSE(W, b, x, y, reg):
    N, M= x.shape   # 3500x784
    mse = la.norm(np.dot(x, W) + b - y) ** 2
    total_loss = 1 / (2 * N) * mse + reg / 2 * (la.norm(W) ** 2)
    return total_loss


def gradMSE(W, b, x, y, reg):
    N, M= x.shape   # 3500x784
    grad_term = np.dot(x, W) + b - y
    gradMSE_bias = 1/N * np.sum(grad_term)
    gradMSE_W = 1/N * np.dot(x.T, grad_term) + reg * W
    
    return gradMSE_bias, gradMSE_W


def crossEntropyLoss(W, b, x, y, reg):
    N, M = x.shape   # 3500x784
    sigmoid = 1/(1 + np.exp(-np.dot(x, W) - b))  #sigmoid function with input Wx+b
    cross_entropy = np.multiply(y, np.log(sigmoid)) + np.multiply(1-y, np.log(1 - sigmoid))
    total_loss = -1/N * np.sum(cross_entropy) + reg / 2 * (la.norm(W) ** 2)
    return total_loss


def gradCE(W, b, x, y, reg):
    N, M = x.shape  # 3500x784
    sigmoid = 1 / (1 + np.exp(-np.dot(x, W) - b))
    gradCE_bias = -1/N * np.sum(y - sigmoid)
    gradCE_W = -1/N * np.dot(x.T, y - sigmoid) + reg * W
    return gradCE_bias, gradCE_W


def grad_descent(W, b, trainingData, trainingLabels, alpha, epochs, reg, EPS, lossType="None"):
    W_comb = []
    b_comb = []

    if lossType == "None":
        print('in GD with \u03B1 = {}, \u03BB = {}'.format(alpha, reg))
        train_W = W
        train_bias = b
        losses = []

        for i in range(epochs):
            gradMSE_bias, gradMSE_W = gradMSE(train_W, train_bias, trainingData, trainingLabels, reg)
            old_W = train_W
            train_W = train_W - alpha * gradMSE_W
            if la.norm(train_W - old_W) < EPS:
                break;
            train_bias = train_bias - alpha * gradMSE_bias
            mse = MSE(train_W, train_bias, trainingData, trainingLabels, reg)
            W_comb.append(train_W)
            b_comb.append(train_bias)
            losses.append(mse)
        plt.plot(losses, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))
        print('GD with \u03B1 = {}, \u03BB = {} finished'.format(alpha, reg))
    else:
        b_comb, W_comb = grad_descent_CE(W, b, trainingData, trainingLabels, alpha, epochs, reg, EPS)

    return b_comb, W_comb


def grad_descent_CE(W, b, train_dt_reshape, trainingLabels, alpha, epochs, reg, EPS):
    print('in GDCE with \u03B1 = {}, \u03BB = {}'.format(alpha, reg))
    train_W = W
    train_bias = b
    W_comb = []
    b_comb = []
    losses = []

    for i in range(epochs):
        gradCE_bias, gradCE_W = gradCE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        old_W = train_W
        train_W = train_W - alpha * gradCE_W
        if la.norm(train_W - old_W) < EPS:
            break;
        train_bias = train_bias - alpha * gradCE_bias
        ce = crossEntropyLoss(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        W_comb.append(train_W)
        b_comb.append(train_bias)
        losses.append(ce)
    plt.plot(losses, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))
    print('GDCE with \u03B1 = {}, \u03BB = {} finished'.format(alpha, reg))
    return b_comb, W_comb


def valid_or_test_loss(Wl, bl, x, y, reg, alpha, lossType="None"):
    print('in calculating validation/test loss with \u03B1 = {}, \u03BB = {}'.format(alpha, reg))
    losses = []

    for i in range(len(Wl)):
        if lossType == "None":
            mse = MSE(Wl[i], bl[i], x, y, reg)
            losses.append(mse)
        else:
            ce = crossEntropyLoss(Wl[i], bl[i], x, y, reg)
            losses.append(ce)

    plt.plot(losses, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))
    print('validation/test loss with \u03B1 = {}, \u03BB = {} finished'.format(alpha, reg))
    return


def accuracy_plot(Wl, bl, x, y, alpha, reg):
    # print('in plotting accuracy')
    accuracy = []

    for i in range(len(Wl)):
        acc = accuracy_calculator(Wl[i], bl[i], x, y)
        accuracy.append(acc)
    plt.plot(accuracy, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))

    # print('plotting accuracy finished')
    return accuracy[len(accuracy)-1]


def accuracy_calculator(W, b, x, y):
    # print('in calculating accuracy')
    y_hat = np.dot(x, W) + b
    accuracy = np.sum((y_hat >= 0.5) == y) / y.shape[0]
    # print('calculating accuracy finished')
    return accuracy


def buildGraph(loss="None"):
    W = tf.Variable(tf.random.truncated_normal([784, 1], stddev=0.5, dtype=tf.float32))
    b = tf.Variable(tf.random.truncated_normal([1, 1], stddev=0.5, dtype=tf.float32))

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 1])
    lambda_ = tf.placeholder(tf.float32)
    tf.set_random_seed(421)

    if loss == "MSE":
        y_hat = tf.matmul(x, W) + b
        loss_t = 0.5 * tf.reduce_mean(tf.square(y - y_hat)) + lambda_ * tf.nn.l2_loss(W)
    elif loss == "CE":
        logits = (tf.matmul(x, W) + b)
        loss_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) + lambda_ * tf.nn.l2_loss(W)

    adam_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_t)
    return x, y, W, b, lambda_, loss_t, adam_op
