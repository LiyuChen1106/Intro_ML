import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from starter import *
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget) # Nx10
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)
N, F = trainData.shape
alpha = 1e-7
gamma = 0.99
epochs = 200

###########################################################
###     data: training input                            ###
###     target: given target                            ###
###     Sh: output from the hidden layer                ###
###     So: output from the outer layer (prediction)    ###
###     K: number of nodes in hidden layer              ###
###     F: number of features                           ###
###########################################################

def gradChecking():
    return

def NN(K):
    W0 = np.random.normal(0, 2 / (K + 10), (K, 10))
    b0 = np.zeros((1, 10))

    Wh = np.random.normal(0, 2 / (F + K), (F, K))
    bh = np.zeros((1, K))

    V_W0 = np.full((K, 10), 1e-5)
    V_Wh = np.full((F, K), 1e-5)
    V_b0 = np.full((1, 10), 1e-5)
    V_bh = np.full((1, K), 1e-5)

    for i in range(epochs):
        # Forward prop
        z_h = computeLayer(trainData, Wh, bh)
        s_h = relu(z_h)
        z_o = computeLayer(s_h, W0, b0)
        s_o = softmax(z_o)

        # Back prop
        grad_CE_Z0 = gradCE(trainTarget, s_o)
        grad_b0 = grad_CE_Z0
        grad_W0 = np.dot(s_h.T, grad_CE_Z0)

        grad_Sh_Zh = s_h
        grad_Sh_Zh[grad_Sh_Zh < 0] = 0
        grad_Sh_Zh[grad_Sh_Zh > 0] = 1
        grad_CE_Sh = np.dot(grad_CE_Z0, W0.T)
        grad_bh = np.dot(grad_CE_Sh, grad_Sh_Zh)
        grad_Wh = np.dot(trainData.T, grad_bh)

        ## update weight
        V_W0 = gamma * V_W0 + alpha * grad_W0
        W0 = W0 - V_W0
        V_Wh = gamma * V_Wh + alpha * grad_Wh
        Wh = Wh - V_Wh

        ## update bias
        V_b0 = gamma * V_b0 + alpha * grad_b0
        b0 = b0 - V_b0
        V_bh = gamma * V_bh + alpha * grad_bh
        bh = bh - V_bh
