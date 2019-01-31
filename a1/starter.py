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
    # Input x should be N(=3600) vectors
    N, M= x.shape  # 3500x784
    mse = la.norm(np.dot(x, W) + b - y) ** 2
    total_loss = 1 / (2 * N) * mse + reg / 2 * la.norm(W) ** 2
    return total_loss

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    N, M= x.shape
    grad_term = np.dot(x, W) + b - y
    gradMSE_bias = 1/N * np.sum(grad_term)
    gradMSE_W = 1/N * np.dot(x.T, grad_term) + reg * W

    return gradMSE_bias, gradMSE_W

# def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here

# def gradCE(W, b, x, y, reg):
    # Your implementation here

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    # Your implementation here
    train_dt_reshape = trainingData.reshape(3500, -1)
    train_W = W
    train_bias = b
    losses = []

    for i in range(iterations):
        gradMSE_bias, gradMSE_W = gradMSE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        old_W = train_W
        train_W = train_W - alpha * gradMSE_W
        if la.norm(train_W - old_W) < EPS:
            break;
        train_bias = train_bias - alpha * gradMSE_bias
        mse = MSE(train_W, train_bias, train_dt_reshape, trainingLabels, reg)
        print(mse)
        losses.append(mse)

    return train_bias, train_W

W = np.zeros((784,1))
b = 0
trainData, _,_,trainTarget,_,_ = loadData()
train_bias, train_W = grad_descent(W,b,trainData,trainTarget,0.005,5000,0,1e-7)

print("-------------bias--------------")
print(train_bias)
print("-------------W-----------------")
print(train_W)


# def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here

