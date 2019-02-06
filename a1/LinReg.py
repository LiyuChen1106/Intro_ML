import matplotlib.pyplot as plt
import numpy as np
from starter import loadData, grad_descent, accuracy_plot, valid_or_test_loss
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)
W = np.random.normal(0, 0.5, (784, 1)) # mean: 0, standard dev: 0.5
b = np.random.normal(0, 0.5, 1)


def validate_test_all_alpha_plot(train_W1, train_bias1, train_W2, train_bias2, train_W3, train_bias3, data, target, suptitle):
    reg = 0
    alpha1 = 0.005
    alpha2 = 0.001
    alpha3 = 0.0001
    #######################Calculate loss
    plt.suptitle(suptitle)
    plt.subplot(121)
    valid_or_test_loss(train_W1, train_bias1, data, target, reg, alpha1)
    valid_or_test_loss(train_W2, train_bias2, data, target, reg, alpha2)
    valid_or_test_loss(train_W3, train_bias3, data, target, reg, alpha3)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    #######################Calculate accuracy
    plt.subplot(122)
    print('final accuracy on validation data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, data, target, alpha1, reg))
    print('final accuracy on validation data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, data, target, alpha2, reg))
    print('final accuracy on validation data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, data, target, alpha3, reg))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()


def linear_reg():
    ###############################################
    ###               1.3                       ###
    ###############################################
    epochs = 5000
    reg = 0
    error_tol = 1e-7

    #######################Train
    #traing losses with different alpha
    plt.figure(1)
    plt.suptitle('Training data')
    plt.subplot(121)
    # alpha = 0.005
    alpha1 = 0.005
    train_bias1, train_W1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol)
    # alpha = 0.001
    alpha2 = 0.001
    train_bias2, train_W2 = grad_descent(W, b, trainData, trainTarget, alpha2, epochs, reg, error_tol)
    # alpha = 0.0001
    alpha3 = 0.0001
    train_bias3, train_W3 = grad_descent(W, b, trainData, trainTarget, alpha3, epochs, reg, error_tol)
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    #######################Calculate accuracy on train
    plt.subplot(122)
    # alpha = 0.005
    print('final accuracy on training data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, trainData, trainTarget, alpha1, reg))
    # alpha = 0.001
    print('final accuracy on training data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, trainData, trainTarget, alpha2, reg))
    # alpha = 0.0001
    print('final accuracy on training data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, trainData, trainTarget, alpha3, reg))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.savefig('train_accuracy_linear.png')

    plt.figure(2)
    validate_test_all_alpha_plot(train_W1, train_bias1, train_W2, train_bias2, train_W3, train_bias3, validData, validTarget,
                                 'Validation data')

    plt.figure(3)
    validate_test_all_alpha_plot(train_W1, train_bias1, train_W2, train_bias2, train_W3, train_bias3, testData, testTarget,
                                 'Test data')

    plt.grid()
    plt.show()

    ###############################################
    ###               1.4                       ###
    ###############################################
    epochs = 5000
    alpha = 0.005
    reg = [0.001, 0.1, 0.5]

linear_reg()