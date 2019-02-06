import matplotlib.pyplot as plt
import numpy as np
from starter import loadData, grad_descent, accuracy_plot, valid_or_test_loss
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)
W = np.random.normal(0, 0.5, (784, 1))  # mean: 0, standard dev: 0.5
b = np.random.normal(0, 0.5, 1)

def log_reg():
    ###############################################
    ###               2.2                       ###
    ###############################################
    epochs = 5000
    reg = 0.1
    error_tol = 1e-7

    #######################Train
    # traing losses with different alpha
    plt.figure(1)
    plt.suptitle('Training data')
    plt.subplot(121)
    # alpha = 0.005
    alpha1 = 0.005
    train_bias1, train_W1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol, "Cross_Entropy")
    plt.xlabel('epochs')
    plt.ylabel('losses')
    #######################Calculate accuracy on train
    plt.subplot(122)
    print('final accuracy on training data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, trainData, trainTarget, alpha1, reg))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()

    #######################Calculate loss on validation
    # validation losses with different alpha
    plt.figure(2)
    plt.suptitle('Validation data')
    plt.subplot(121)
    # alpha = 0.005
    valid_or_test_loss(train_W1, train_bias1, validData, validTarget, reg, alpha1, "Cross_Entropy")
    plt.xlabel('epochs')
    plt.ylabel('losses')
    #######################Calculate accuracy on validation
    # valid
    plt.subplot(122)
    print('final accuracy on validation data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, validData, validTarget, alpha1, reg))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()

    #######################Calculate loss on test
    # test losses with different alpha
    plt.figure(3)
    plt.suptitle('Test data')
    plt.subplot(121)
    # alpha = 0.005
    valid_or_test_loss(train_W1, train_bias1, testData, testTarget, reg, alpha1, "Cross_Entropy")
    plt.xlabel('epochs')
    plt.ylabel('losses')
    #######################Calculate accuracy on test
    # test
    plt.subplot(122)
    print('final accuracy on test data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, testData, testTarget, alpha1, reg))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.grid()
    plt.show()


log_reg()