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
    losses1 = valid_or_test_loss(train_W1, train_bias1, data, target, reg, alpha1)
    losses2 = valid_or_test_loss(train_W2, train_bias2, data, target, reg, alpha2)
    losses3 = valid_or_test_loss(train_W3, train_bias3, data, target, reg, alpha3)
    plt.plot(losses1, label='\u03B1 = {}, \u03BB = {}'.format(alpha1, reg))
    plt.plot(losses2, label='\u03B1 = {}, \u03BB = {}'.format(alpha2, reg))
    plt.plot(losses3, label='\u03B1 = {}, \u03BB = {}'.format(alpha3, reg))
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.grid()
    #######################Calculate accuracy
    print('final accuracy on validation data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, data, target, alpha1, reg))
    print('final accuracy on validation data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, data, target, alpha2, reg))
    print('final accuracy on validation data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, data, target, alpha3, reg))


def normal_equation(x, y, reg):
   X=x
   N=y.shape[0]
   d=x.shape[1]
   x_zero=np.ones((N,1))
   I=np.identity(d+1)
   I[0,0]=0
   X=np.append(x_zero,X,axis=1)
   print(X.shape)
   W_star=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+reg*I),X.T),y)
   bias=W_star[0]
   W_star=np.delete(W_star,(0),axis=0)
   return W_star, bias

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
    # alpha = 0.005
    alpha1 = 0.005
    train_bias1, train_W1, loss_a1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg, error_tol)
    # alpha = 0.001
    alpha2 = 0.001
    train_bias2, train_W2, loss_a2 = grad_descent(W, b, trainData, trainTarget, alpha2, epochs, reg, error_tol)
    # alpha = 0.0001
    alpha3 = 0.0001
    train_bias3, train_W3, loss_a3 = grad_descent(W, b, trainData, trainTarget, alpha3, epochs, reg, error_tol)
    plt.plot(loss_a1, label='MSE: \u03B1 = {}, \u03BB = {}'.format(alpha1, reg))
    plt.plot(loss_a2, label='MSE: \u03B1 = {}, \u03BB = {}'.format(alpha2, reg))
    plt.plot(loss_a3, label='MSE: \u03B1 = {}, \u03BB = {}'.format(alpha3, reg))
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.savefig('train_loss_linear.png')
    #######################Calculate accuracy on train
    # alpha = 0.005
    print('final accuracy on training data (alpha = 0.005):')
    print(accuracy_plot(train_W1, train_bias1, trainData, trainTarget, alpha1, reg))
    # alpha = 0.001
    print('final accuracy on training data (alpha = 0.001):')
    print(accuracy_plot(train_W2, train_bias2, trainData, trainTarget, alpha2, reg))
    # alpha = 0.0001
    print('final accuracy on training data (alpha = 0.0001):')
    print(accuracy_plot(train_W3, train_bias3, trainData, trainTarget, alpha3, reg))


    plt.figure(2)
    validate_test_all_alpha_plot(train_W1, train_bias1, train_W2, train_bias2, train_W3, train_bias3, validData, validTarget,
                                 'Validation data')
    plt.savefig('valid_loss_linear.png')

    plt.figure(3)
    validate_test_all_alpha_plot(train_W1, train_bias1, train_W2, train_bias2, train_W3, train_bias3, testData, testTarget,
                                 'Test data')
    plt.savefig('test_loss_linear.png')

    ###############################################
    ###               1.4                       ###
    ###############################################
    epochs = 5000
    alpha = 0.005
    reg = [0.001, 0.1, 0.5]
    error_tol = 1e-7

    plt.figure(4)
    plt.suptitle('Training data')
    train_bias4, train_W4, loss_r1 = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg[0], error_tol)
    train_bias5, train_W5, loss_r2 = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg[1], error_tol)
    train_bias6, train_W6, loss_r3 = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg[2], error_tol)
    plt.plot(loss_r1, label='MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg[0]))
    plt.plot(loss_r2, label='MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg[1]))
    plt.plot(loss_r3, label='MSE: \u03B1 = {}, \u03BB = {}'.format(alpha, reg[2]))
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.grid()
    plt.savefig('train_loss_linear_reg.png')
    print('final accuracy on training data (reg = {}):'.format(reg[0]))
    print(accuracy_plot(train_W4, train_bias4, trainData, trainTarget, alpha, reg[0]))
    print('final accuracy on training data (reg = {}):'.format(reg[1]))
    print(accuracy_plot(train_W5, train_bias5, trainData, trainTarget, alpha, reg[1]))
    print('final accuracy on training data (reg = {}):'.format(reg[2]))
    print(accuracy_plot(train_W6, train_bias6, trainData, trainTarget, alpha, reg[2]))

    ###VALID
    #######################Calculate loss
    plt.figure(5)
    plt.suptitle('Validation data')
    losses1 = valid_or_test_loss(train_W4, train_bias4, validData, validTarget, reg[0], alpha)
    losses2 = valid_or_test_loss(train_W5, train_bias5, validData, validTarget, reg[1], alpha)
    losses3 = valid_or_test_loss(train_W6, train_bias6, validData, validTarget, reg[2], alpha)
    plt.plot(losses1, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg[0]))
    plt.plot(losses2, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg[1]))
    plt.plot(losses3, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg[2]))
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.grid()
    plt.savefig('valid_loss_linear_reg.png')
    #######################Calculate accuracy
    print('final accuracy on validation data (reg = {}):'.format(reg[0]))
    print(accuracy_plot(train_W4, train_bias4, validData, validTarget, alpha, reg[0]))
    print('final accuracy on validation data (reg = {}):'.format(reg[1]))
    print(accuracy_plot(train_W5, train_bias5, validData, validTarget, alpha, reg[1]))
    print('final accuracy on validation data (reg = {}):'.format(reg[2]))
    print(accuracy_plot(train_W6, train_bias6, validData, validTarget, alpha, reg[2]))

    ###TEST
    #######################Calculate loss
    plt.figure(6)
    plt.suptitle('Test data')
    losses4 = valid_or_test_loss(train_W4, train_bias4, testData, testTarget, reg[0], alpha)
    losses5 = valid_or_test_loss(train_W5, train_bias5, testData, testTarget, reg[1], alpha)
    losses6 = valid_or_test_loss(train_W6, train_bias6, testData, testTarget, reg[2], alpha)
    plt.plot(losses4, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg[0]))
    plt.plot(losses5, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg[1]))
    plt.plot(losses6, label='\u03B1 = {}, \u03BB = {}'.format(alpha, reg[2]))
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()
    plt.grid()
    plt.savefig('test_loss_linear_reg.png')
    #######################Calculate accuracy
    print('final accuracy on validation data (reg = {}):'.format(reg[0]))
    print(accuracy_plot(train_W4, train_bias4, testData, testTarget, alpha, reg[0]))
    print('final accuracy on validation data (reg = {}):'.format(reg[1]))
    print(accuracy_plot(train_W5, train_bias5, testData, testTarget, alpha, reg[1]))
    print('final accuracy on validation data (reg = {}):'.format(reg[2]))
    print(accuracy_plot(train_W6, train_bias6, testData, testTarget, alpha, reg[2]))

    plt.show()

linear_reg()