import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from starter import loadData, buildGraph
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)

def SGD():
    ###############################################
    ###               3.2                       ###
    ###############################################

    minibatch(500, trainData, trainTarget, lossType="MSE")

    return


def minibatch(minibatch_size, trainingData, trainingTarget, lossType):
    N, _ = trainingData.shape
    x, y, W, b, lambda_, loss_t, adam_op = buildGraph(loss=lossType)

    n_epochs = 700
    iterations = N // minibatch_size

    # "minibatch" training
    acc_log = np.zeros(n_epochs)
    loss_log = np.zeros(n_epochs)

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
        acc_log[i] = np.sum((np.dot(trainingData, train_W) + train_b >= 0.5) == trainingTarget) / trainingTarget.shape[0]
        loss_log[i] = sess.run(loss_t, feed_dict={x: trainingData, y: trainingTarget, lambda_: 0})


    # Canvas making
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), sharey=False, facecolor='w')
    f.suptitle('Minibatch size {}'.format(minibatch_size))

    # loss plot
    ax0.set_ylabel("loss")
    ax0.set_xlabel("iterations")
    # accuracy plot
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("iterations")


    ax0.plot(loss_log)
    ax1.plot(acc_log)

    ax0.grid()
    ax1.grid()
    # ax0.legend(loc=4, fontsize=16)
    # ax1.legend(loc=3, fontsize=16)
    plt.show()


SGD()