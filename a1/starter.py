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
    train_dt_reshape = trainingData.reshape(3500, -1)
    W_comb = []
    b_comb = []

    if lossType == "None":
        print('in GD with \u03B1 = {}, \u03BB = {}'.format(alpha, reg))
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
            W_comb.append(train_W)
            b_comb.append(train_bias)
            losses.append(mse)
        plt.plot(losses, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))
        print('GD with \u03B1 = {}, \u03BB = {} finished'.format(alpha, reg))
    else:
        b_comb, W_comb = grad_descent_CE(W, b, train_dt_reshape, trainingLabels, alpha, epochs, reg, EPS)

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
    N, _, _ = x.shape
    x_reshape = x.reshape(N, -1)
    losses = []

    for i in range(len(Wl)):
        if lossType == "None":
            mse = MSE(Wl[i], bl[i], x_reshape, y, reg)
            losses.append(mse)
        else:
            ce = crossEntropyLoss(Wl[i], bl[i], x_reshape, y, reg)
            losses.append(ce)

    plt.plot(losses, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))
    print('validation/test loss with \u03B1 = {}, \u03BB = {} finished'.format(alpha, reg))
    return

def accuracy_plot(Wl, bl, x, y, alpha, reg):
    # print('in plotting accuracy')
    N, _, _ = x.shape
    x_reshape = x.reshape(N, -1)
    accuracy = []

    for i in range(len(Wl)):
        acc = accuracy_calculator(Wl[i], bl[i], x_reshape, y)
        accuracy.append(acc)
    plt.plot(accuracy, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg))

    # print('plotting accuracy finished')
    return accuracy[len(accuracy)-1]

def accuracy_calculator(W, b, x, y):
    # print('in calculating accuracy')
    y_hat = np.sign(np.dot(x, W) - b)
    E = y - y_hat
    accuracy = np.count_nonzero(E == 1) + np.count_nonzero(E == 0)
    # print('calculating accuracy finished')
    return accuracy / y.shape[0]


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
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
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
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()


    ###############################################
    ###               1.3                       ###
    ###############################################
    epochs = 5000
    reg = 0
    error_tol = 1e-7

    #######################Train
    #traing losses with different alpha
    plt.figure(1)
    # alpha = 0.005
    alpha1 = 0.005
    train_bias1, train_W1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol)
    # alpha = 0.001
    alpha2 = 0.001
    train_bias2, train_W2 = grad_descent(W, b, trainData, trainTarget, alpha2, epochs, reg, error_tol)
    # alpha = 0.0001
    alpha3 = 0.0001
    train_bias3, train_W3 = grad_descent(W, b, trainData, trainTarget, alpha3, epochs, reg, error_tol)
    plt.title('GD linear regression training loss')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig('train_loss_linear.png')

    #######################Calculate loss on validation and test
    # validation losses with different alpha
    plt.figure(2)
    # alpha = 0.005
    valid_or_test_loss(train_W1, train_bias1, validData, validTarget, reg, alpha1)
    # alpha = 0.001
    valid_or_test_loss(train_W2, train_bias2, validData, validTarget, reg, alpha2)
    # alpha = 0.0001
    valid_or_test_loss(train_W3, train_bias3, validData, validTarget, reg, alpha3)
    plt.title('GD linear regression validation loss')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig('valid_loss_linear.png')

    # test losses with different alpha
    plt.figure(3)
    # alpha = 0.005
    valid_or_test_loss(train_W1, train_bias1, testData, testTarget, reg, alpha1)
    # alpha = 0.001
    valid_or_test_loss(train_W2, train_bias2, testData, testTarget, reg, alpha2)
    # alpha = 0.0001
    valid_or_test_loss(train_W3, train_bias3, testData, testTarget, reg, alpha3)
    plt.title('GD linear regression test loss')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig('test_loss_linear.png')

    #######################Calculate accuracy on train, validation and test
    # Final accuracy
    plt.figure(4)
    # alpha = 0.005
    print('final accuracy on training data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, trainData, trainTarget, alpha1, reg))
    # alpha = 0.001
    print('final accuracy on training data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, trainData, trainTarget, alpha2, reg))
    # alpha = 0.0001
    print('final accuracy on training data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, trainData, trainTarget, alpha3, reg))
    plt.title('Accuracy on training data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('train_accuracy_linear.png')

    plt.figure(5)
    # alpha = 0.005
    print('final accuracy on validation data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, validData, validTarget, alpha1, reg))
    # alpha = 0.001
    print('final accuracy on validation data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, validData, validTarget, alpha2, reg))
    # alpha = 0.0001
    print('final accuracy on validation data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, validData, validTarget, alpha3, reg))
    plt.title('Accuracy on validation data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('valid_accuracy_linear.png')

    plt.figure(6)
    # alpha = 0.005
    print('final accuracy on test data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, testData, testTarget, alpha1, reg))
    # alpha = 0.001
    print('final accuracy on test data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, testData, testTarget, alpha2, reg))
    # alpha = 0.0001
    print('final accuracy on test data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, testData, testTarget, alpha3, reg))
    plt.title('Accuracy on test data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('test_accuracy_linear.png')
    plt.show()

    ###############################################
    ###               1.4                       ###
    ###############################################
    epochs = 5000
    alpha = 0.005
    reg = [0.001, 0.1, 0.5]


def log_reg():
    # part 2.2
    W = np.zeros((784, 1))
    b = 0
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()


    ###############################################
    ###               2.2                       ###
    ###############################################
    epochs = 5000
    reg = 0.1
    error_tol = 1e-7

    #######################Train
    # traing losses with different alpha
    plt.figure(1)
    plt.subplot(131)
    # alpha = 0.005
    alpha1 = 0.005
    train_bias1, train_W1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol, "Cross_Entropy")
    plt.title('GDCE logistic regression training loss')
    plt.xlabel('epochs')
    plt.ylabel('losses')

    #######################Calculate loss on validation and test
    # validation losses with different alpha
    plt.subplot(132)
    # alpha = 0.005
    valid_or_test_loss(train_W1, train_bias1, validData, validTarget, reg, alpha1, "Cross_Entropy")
    plt.title('GDCE logistic regression validation loss')
    plt.xlabel('epochs')
    plt.ylabel('losses')

    # test losses with different alpha
    plt.subplot(133)
    # alpha = 0.005
    valid_or_test_loss(train_W1, train_bias1, testData, testTarget, reg, alpha1, "Cross_Entropy")
    plt.title('GDCE logistic regression test loss')
    plt.xlabel('epochs')
    plt.ylabel('losses')

    #######################Calculate accuracy on train, validation and test
    # Final accuracy
    plt.figure(2)
    # train
    plt.subplot(131)
    print('final accuracy on training data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, trainData, trainTarget, alpha1, reg))
    plt.title('Accuracy on training data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    # valid
    plt.subplot(132)
    print('final accuracy on validation data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, validData, validTarget, alpha1, reg))
    plt.title('Accuracy on validation data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    # test
    plt.subplot(133)
    print('final accuracy on test data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, testData, testTarget, alpha1, reg))
    plt.title('Accuracy on test data')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


# linear_reg()
log_reg()