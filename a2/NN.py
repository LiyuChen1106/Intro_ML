import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from starter import *
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrainTarget, newvalidTarget, newtestTarget = convertOneHot(trainTarget, validTarget, testTarget) # Nx10
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)
N, F = trainData.shape
alpha = 1e-7
gamma = 0.99

###########################################################
###     data: training input                            ###
###     target: given target                            ###
###     Sh: output from the hidden layer                ###
###     So: output from the outer layer (prediction)    ###
###     K: number of nodes in hidden layer              ###
###     F: number of features                           ###
###########################################################
def grad_outer(s_o, s_h):
    grad_CE_Z0 = gradCE(newtrainTarget, s_o)
    #weight
    grad_W0 = np.matmul(np.transpose(s_h), grad_CE_Z0)
    #bias
    init_ones = np.ones((1, N))
    grad_b0 = np.matmul(init_ones, grad_CE_Z0)

    return grad_W0, grad_b0

def grad_hidden(s_o, z_h, W0):
    grad_CE_Z0 = gradCE(newtrainTarget, s_o)
    grad_Sh_Zh = z_h
    grad_Sh_Zh[grad_Sh_Zh <= 0] = 0
    grad_Sh_Zh[grad_Sh_Zh > 0] = 1
    grad_CE_Sh = np.matmul(grad_CE_Z0, np.transpose(W0))
    grad_CE_Zh = grad_CE_Sh * grad_Sh_Zh
    # weight
    grad_Wh = np.matmul(np.transpose(trainData), grad_CE_Zh)
    #bias
    init_ones = np.ones((1, N))
    grad_bh = np.matmul(init_ones, grad_CE_Zh)

    return grad_Wh, grad_bh

def gradChecking():
    return

def full_forward_propagation(data, target, W0, Wh, b0, bh):
    # Forward prop
    z_h = computeLayer(data, Wh, bh)
    s_h = relu(z_h)
    z_o = computeLayer(s_h, W0, b0)
    s_o = softmax(z_o)

    # prediction and accuracy calculation
    prediction = np.argmax(s_o, axis=1)
    target = np.argmax(target, axis=1)

    acc = np.sum(prediction == target) / data.shape[0]

    return z_h, s_h, z_o, s_o, acc

def NN(K, ax=None, epochs=200):
    W0 = np.random.normal(0, np.sqrt(2 / (K + 10)), (K, 10))
    b0 = np.zeros((1, 10))

    Wh = np.random.normal(0, np.sqrt(2 / (F + K)), (F, K))
    bh = np.zeros((1, K))

    V_W0 = np.full((K, 10), 1e-5)
    V_Wh = np.full((F, K), 1e-5)
    V_b0 = np.zeros((1, 10))
    V_bh = np.zeros((1, K))

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    # print('final accuracy on training data:')

    for i in range(epochs):
        # Forward prop
        z_h, s_h, z_o, s_o, train_acc = full_forward_propagation(trainData, newtrainTarget, W0, Wh, b0, bh)
        train_accuracy.append(train_acc)
        _, _, _, _, valid_acc = full_forward_propagation(validData, newvalidTarget, W0, Wh, b0, bh)
        valid_accuracy.append(valid_acc)
        _, _, _, _, test_acc = full_forward_propagation(testData, newtestTarget, W0, Wh, b0, bh)
        test_accuracy.append(test_acc)

        # print(train_acc)

        # Back prop
        grad_W0, grad_b0 = grad_outer(s_o, s_h)
        grad_Wh, grad_bh = grad_hidden(s_o, z_h, W0)

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

    print('hidden nodes={}, epochs={}--------------------------'.format(K, epochs))
    print('final accuracy on training data:')
    print(train_accuracy[len(train_accuracy) - 1])
    print('final accuracy on valid data:')
    print(valid_accuracy[len(valid_accuracy) - 1])
    print('final accuracy on test data :')
    print(test_accuracy[len(test_accuracy) - 1])

    #plot
    if ax is None:
        plt.plot(train_accuracy, label='training')
        plt.plot(valid_accuracy, label='validation')
        plt.plot(test_accuracy, label='test')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.grid()
    else:
        ax.plot(train_accuracy, label='training')
        ax.plot(valid_accuracy, label='validation')
        ax.plot(test_accuracy, label='test')
        ax.set_xlabel('epochs')
        ax.set_ylabel('accuracy')
        ax.legend()
        ax.grid()


###############################################
###               1.3                       ###
###############################################
# plt.figure(1, figsize=(6, 4))
# NN(1000, None)
# plt.title("accuracy curves: hidden nodes=1000")
# plt.savefig('images/hidden_1000.png', dpi=1200)

###############################################
###               1.4                       ###
###############################################
# plt.figure(2, figsize=(16, 4))
# ax1 = plt.subplot("131")
# ax1.set_title("hidden nodes=100")
# NN(100, ax1)
# ax2 = plt.subplot("132")
# ax2.set_title("hidden nodes=500")
# NN(500, ax2)
# ax3 = plt.subplot("133")
# ax3.set_title("hidden nodes=2000")
# NN(2000, ax3)
# plt.savefig('images/hidden_all.png', dpi=1200)

plt.figure(3, figsize=(6, 4))
plt.xlim(100, 600)
plt.ylim(0.85, 0.93)
NN(1000, None, epochs=600)
plt.title("hidden nodes=1000, epochs=600")
plt.savefig('images/early stop.png', dpi=1000)

plt.show()
