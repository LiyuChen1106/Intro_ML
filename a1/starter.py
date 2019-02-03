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
    # print("x dot W------------------------------")
    # print(np.dot(x,W))
    # print("y mult x------------------------------")
    # print(np.multiply(y, x))
    # print("sigmoid------------------------------")
    # print(sigmoid)
    gradCE_W = -1/N * np.dot(x.T, y - sigmoid) + reg * W
    # print("gradCE_W------------------------------")
    # print(gradCE_W)
    return gradCE_bias, gradCE_W


def grad_descent(W, b, trainingData, trainingLabels, alpha, epochs, reg, EPS):
    # Your implementation here
    train_dt_reshape = trainingData.reshape(3500, -1)
    train_W = W
    train_bias = b
    losses = []
    
    for i in range(epochs):
        gradMSE_bias, gradMSE_W = gradMSE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        old_W = train_W
        train_W = train_W - alpha * gradMSE_W
        if la.norm(train_W - old_W) < EPS:
            break;
        train_bias = train_bias - alpha * gradMSE_bias
        mse = MSE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        print(i)
        print(mse)
        losses.append(mse)
    plt.figure()
    plt.plot(losses)
    plt.show()

    return train_bias, train_W


def grad_descent_CE(W, b, trainingData, trainingLabels, alpha, epochs, reg, EPS):
    # Your implementation here
    train_dt_reshape = trainingData.reshape(3500, -1)
    train_W = W
    train_bias = b
    losses = []

    for i in range(epochs):
        gradCE_bias, gradCE_W = gradCE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        old_W = train_W
        train_W = train_W - alpha * gradCE_W
        if la.norm(train_W - old_W) < EPS:
            break;
        train_bias = train_bias - alpha * gradCE_bias
        ce = crossEntropyLoss(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        print(i)
        print(ce)
        losses.append(ce)
    plt.figure()
    plt.plot(losses)
    plt.show()

    return train_bias, train_W

"""def buildGraph(loss=None):
    #Initialize weight and bias tensors
    w_tensor = tf.random.truncated_normal(
        shape,
        mean=0.0,
        stddev=0.5,
        dtype=tf.float32,
        seed=None,
        name=None
    )

    tf.placeholder(dtype, shape=None, name=None)
    tf.set_random_seed(421)
    if loss == "MSE":
    # Your implementation
    elif loss == "CE":"""
# Your implementation here


def general_test():
    # general test
    W = np.zeros((784, 1))
    b = 0
    trainData, _, _, trainTarget, _, _ = loadData()
    red = trainData.reshape(3500, -1)
    lossmse = MSE(W, b, red, trainTarget, 0)
    gradmseb, gradmseW = gradMSE(W, b, red, trainTarget, 0)
    lossce = crossEntropyLoss(W, b, red, trainTarget, 0)
    gradceb, gradceW = gradCE(W, b, red, trainTarget, 0)
    print("mse loss------------------------------")
    print(lossmse)
    print("grad mse------------------------------")
    print(gradmseb)
    print(gradmseW)
    print("ce loss------------------------------")
    print(lossce)
    print("grad ce------------------------------")
    print(gradceb)
    print(gradceW)


def linear_reg():
    # part 1.3
    W = np.zeros((784, 1))
    b = 0
    trainData, _, _, trainTarget, _, _ = loadData()
    alpha = 0.005
    epochs = 5000
    reg = 0
    error_tol = 1e-7
    train_bias, train_W = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol)

    print("-------------bias--------------")
    print(train_bias)
    print("-------------W-----------------")
    print(train_W)

    # alpha = 0.001
    alpha = 0.001
    train_bias, train_W = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol)

    print("-------------bias--------------")
    print(train_bias)
    print("-------------W-----------------")
    print(train_W)

    # alpha = 0.001
    alpha = 0.001
    train_bias, train_W = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol)

    print("-------------bias--------------")
    print(train_bias)
    print("-------------W-----------------")
    print(train_W)

    # alpha = 0.0001
    alpha = 0.0001
    train_bias, train_W = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol)

    print("-------------bias--------------")
    print(train_bias)
    print("-------------W-----------------")
    print(train_W)

    # part 1.4
    epochs = 5000
    alpha = 0.005
    reg = [0.001, 0.1, 0.5]


def log_reg():
    # part 2.2
    """def grad_descent(W, b, x, y, alpha, epochs, reg, EPS, lossType="None"):
        # Your implementation here
        train_dt_reshape = trainingData.reshape(3500, -1)
        train_W = W
        train_bias = b
        losses = []

        for i in range(epochs):
            gradCE_bias, gradCE_W = gradMSE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
            old_W = train_W
            train_W = train_W - alpha * gradMSE_W
            if la.norm(train_W - old_W) < EPS:
                break;
            train_bias = train_bias - alpha * gradCE_bias
            ce = crossEntropyLoss(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
            print(ce)
            losses.append(ce)

        return losses
    """
    W = np.zeros((784, 1))
    b = 0
    trainData, _, _, trainTarget, _, _ = loadData()
    alpha = 0.1
    epochs = 5000
    reg = 0
    error_tol = 1e-7
    train_bias, train_W = grad_descent_CE(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol)

general_test()