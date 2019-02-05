import matplotlib.pyplot as plt
import tensorflow as tf
from starter import loadData, buildGraph
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)

def SGD():
    ###############################################
    ###               3.2                       ###
    ###############################################
    n_epochs = 700

    # Create session to execute ops
    sess = tf.InteractiveSession()

    return


def minibatch(minibatch_size, trainingData, trainingTarget):
    N, _ = trainingData
    x, y, W, b, lambda_, loss_t, adam_op = buildGraph(loss="MSE")
    y_hat = tf.matmul(x, W) + b
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Canvas making
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), sharey=False, facecolor='w')
    f.suptitle('Batch {}'.format(minibatch_size))

    # accuracy plot
    ax0.set_ylabel("accuracy")
    ax0.set_xlabel("iterations")
    # loss plot
    ax1.set_ylabel("loss")
    ax1.set_xlabel("iterations")

    # "minibatch" training
    acc_log = tf.zeros(len(image_set))
    loss_log = tf.zeros(len(image_set))

    ax0.plot(acc_log, 'b', label="Minibatch {}".format(minibatch_size))
    ax1.plot(loss_log, 'b', label="Minibatch {}".format(minibatch_size))
