import matplotlib.pyplot as plt
import numpy as np
from starter import loadData, grad_descent, accuracy_plot, accuracy_calc, valid_or_test_loss
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
    plt.figure(1, figsize=(16, 4))
    plt.subplot(121)
    # alpha = 0.005
    alpha1 = 0.005
    train_bias1, train_W1, train_losses = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol, "Cross_Entropy")
    valid_losses = valid_or_test_loss(train_W1, train_bias1, validData, validTarget, reg, alpha1, "Cross_Entropy")
    test_losses = valid_or_test_loss(train_W1, train_bias1, testData, testTarget, reg, alpha1, "Cross_Entropy")
    plt.plot(train_losses, label='training')
    plt.plot(valid_losses, label='validation')
    plt.plot(test_losses, label='test')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.grid()

    #######################Calculate accuracy on train
    plt.subplot(122)
    print('final accuracy on training data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, trainData, trainTarget, alpha1, reg))
    print('final accuracy on validation data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, validData, validTarget, alpha1, reg))
    print('final accuracy on test data (alpha = 0.005, lambda = 0.1):')
    print(accuracy_plot(train_W1, train_bias1, testData, testTarget, alpha1, reg))
    train_acc = accuracy_calc(train_W1, train_bias1, trainData, trainTarget, alpha1, reg)
    valid_acc = accuracy_calc(train_W1, train_bias1, validData, validTarget, alpha1, reg)
    test_acc = accuracy_calc(train_W1, train_bias1, testData, testTarget, alpha1, reg)
    plt.plot(train_acc, label='training')
    plt.plot(valid_acc, label='validation')
    plt.plot(test_acc, label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('log_loss_acc.png')

    ###############################################
    ###               2.3                       ###
    ###############################################
    plt.figure(2, figsize=(16, 4))
    reg = 0
    train_biasu, train_Wu, train_losses_ce = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol, "Cross_Entropy")
    train_biasv, train_Wv, train_losses1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol)
    plt.plot(train_losses_ce, label='CE')
    plt.plot(train_losses1, label='MSE')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.grid()
    plt.savefig('linear_log_compare.png')
    plt.show()


log_reg()