import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from starter import loadData, buildGraph
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)

def SGD():
    beta1_d = 0.9
    beta2_d = 0.999
    epi_d = 1e-8

    ###############################################
    ###               3.2                       ###
    ###############################################
    plt.figure(1)
    minibatch(500, trainData, trainTarget, beta1_d, beta2_d, epi_d, True, lossType="MSE")
    plt.savefig('SGD_MSE_500.png')

    ###############################################
    ###               3.3                       ###
    ###############################################
    plt.figure(2)
    minibatch(100, trainData, trainTarget, beta1_d, beta2_d, epi_d, True, lossType="MSE")
    plt.savefig('SGD_MSE_100.png')
    plt.figure(3)
    minibatch(700, trainData, trainTarget, beta1_d, beta2_d, epi_d, True, lossType="MSE")
    plt.savefig('SGD_MSE_700.png')
    plt.figure(4)
    minibatch(1750, trainData, trainTarget, beta1_d, beta2_d, epi_d, True, lossType="MSE")
    plt.savefig('SGD_MSE_1750.png')

    ###############################################
    ###               3.4                       ###
    ###############################################
    plt.figure(5, figsize=(16, 4))
    plt.subplot(121)
    minibatch(500, trainData, trainTarget, 0.95, beta2_d, epi_d, False, lossType="MSE")
    plt.title("\u03B2_1 = 0.95")
    plt.subplot(122)
    minibatch(500, trainData, trainTarget, 0.99, beta2_d, epi_d, False, lossType="MSE")
    plt.title("\u03B2_1 = 0.99")
    plt.savefig('SGD_MSE_BETA1.png')

    plt.figure(6, figsize=(16, 4))
    plt.subplot(121)
    minibatch(500, trainData, trainTarget, beta1_d, 0.99, epi_d, False, lossType="MSE")
    plt.title("\u03B2_2 = 0.99")
    plt.subplot(122)
    minibatch(500, trainData, trainTarget, beta1_d, 0.9999, epi_d, False, lossType="MSE")
    plt.title("\u03B2_2 = 0.9999")
    plt.savefig('SGD_MSE_BETA2.png')

    plt.figure(7, figsize=(16, 4))
    plt.subplot(121)
    minibatch(500, trainData, trainTarget, beta1_d, beta2_d, 1e-9, False, lossType="MSE")
    plt.title("\u03B5 = 1e-9")
    plt.subplot(122)
    minibatch(500, trainData, trainTarget, beta1_d, beta2_d, 1e-4, False, lossType="MSE")
    plt.title("\u03B5 = 1e-4")
    plt.savefig('SGD_MSE_EPI.png')

    ###############################################
    ###               3.5                       ###
    ###############################################
    # plt.figure(8)
    # minibatch(500, trainData, trainTarget, beta1_d, beta2_d, epi_d, True, lossType="CE")
    # plt.savefig('SGD_CE_500.png')
    #
    # plt.figure(9, figsize=(16, 4))
    # plt.subplot(121)
    # minibatch(500, trainData, trainTarget, 0.95, beta2_d, epi_d, False, lossType="CE")
    # plt.title("\u03B2_1 = 0.95")
    # plt.subplot(122)
    # minibatch(500, trainData, trainTarget, 0.99, beta2_d, epi_d, False, lossType="CE")
    # plt.title("\u03B2_1 = 0.99")
    # plt.savefig('SGD_CE_BETA1.png')
    #
    # plt.figure(10, figsize=(16, 4))
    # plt.subplot(121)
    # minibatch(500, trainData, trainTarget, beta1_d, 0.99, epi_d, False, lossType="CE")
    # plt.title("\u03B2_2 = 0.99")
    # plt.subplot(122)
    # minibatch(500, trainData, trainTarget, beta1_d, 0.9999, epi_d, False, lossType="CE")
    # plt.title("\u03B2_2 = 0.9999")
    # plt.savefig('SGD_CE_BETA2.png')
    #
    # plt.figure(11, figsize=(16, 4))
    # plt.subplot(121)
    # minibatch(500, trainData, trainTarget, beta1_d, beta2_d, 1e-9, False, lossType="CE")
    # plt.title("\u03B5 = 1e-9")
    # plt.subplot(122)
    # minibatch(500, trainData, trainTarget, beta1_d, beta2_d, 1e-4, False, lossType="CE")
    # plt.title("\u03B5 = 1e-4")
    # plt.savefig('SGD_CE_EPI.png')

    plt.show()

    return


def minibatch(minibatch_size, trainingData, trainingTarget, beta1, beta2, epsilon, plot_loss, lossType):
    N, _ = trainingData.shape
    x, y, W, b, lambda_, loss_t, adam_op = buildGraph(beta1, beta2, epsilon, loss=lossType)

    n_epochs = 700
    iterations = N // minibatch_size

    # "minibatch" training
    acc_train = np.zeros(n_epochs)
    loss_train = np.zeros(n_epochs)
    acc_valid = np.zeros(n_epochs)
    loss_valid = np.zeros(n_epochs)
    acc_test = np.zeros(n_epochs)
    loss_test = np.zeros(n_epochs)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()

    sess.run(init)

    for i in range(n_epochs):
        #shuffle
        s = np.arange(N)
        np.random.shuffle(s)
        trainingData = trainingData[s]
        trainingTarget = trainingTarget[s]

        #iterating mini batch
        for j in range(iterations):
            batch_data = trainingData[j*minibatch_size:(j+1)*minibatch_size, :]
            batch_target = trainingTarget[j*minibatch_size:(j+1)*minibatch_size, :]
            _, train_W, train_b = sess.run([adam_op, W, b], feed_dict={x: batch_data, y: batch_target, lambda_: 0})

        #calc accuracy and loss for each epoch
        acc_train[i] = np.sum((np.dot(trainingData, train_W) + train_b >= 0.5) == trainingTarget) / trainingTarget.shape[0]
        loss_train[i] = sess.run(loss_t, feed_dict={x: trainingData, y: trainingTarget, lambda_: 0})
        #valid and test
        acc_valid[i] = np.sum((np.dot(validData, train_W) + train_b >= 0.5) == validTarget) / validTarget.shape[0]
        loss_valid[i] = sess.run(loss_t, feed_dict={x: validData, y: validTarget, lambda_: 0})
        acc_test[i] = np.sum((np.dot(testData, train_W) + train_b >= 0.5) == testTarget) / testTarget.shape[0]
        loss_test[i] = sess.run(loss_t, feed_dict={x: testData, y: testTarget, lambda_: 0})


    # Canvas making
    if(plot_loss):
        f, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), sharey=False, facecolor='w')
        f.suptitle('Minibatch size {}'.format(minibatch_size))

        # loss plot
        ax0.set_ylabel("loss")
        ax0.set_xlabel("iterations")
        # accuracy plot
        ax1.set_ylabel("accuracy")
        ax1.set_xlabel("iterations")


        ax0.plot(loss_train, label='training data')
        ax0.plot(loss_valid, label='validation data')
        ax0.plot(loss_test, label='test data')
        ax1.plot(acc_train, label='training data')
        ax1.plot(acc_valid, label='validation data')
        ax1.plot(acc_test, label='test data')


        ax0.grid()
        ax1.grid()
        ax0.legend(loc=1, fontsize=10)
        ax1.legend(loc=4, fontsize=10)
    else:
        plt.ylabel("accuracy")
        plt.xlabel("iterations")

        plt.plot(acc_train, label='training data')
        plt.plot(acc_valid, label='validation data')
        plt.plot(acc_test, label='test data')

        plt.grid()
        plt.legend(loc=4, fontsize=10)


SGD()