import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from starter import *
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget) # Nx10
N = trainData.shape[0]
V = validData.shape[0]
T = testData.shape[0]
trainData = trainData.reshape(trainData.shape[0], -1)
validData = validData.reshape(validData.shape[0], -1)
testData = testData.reshape(testData.shape[0], -1)

def NNTF(drop_put, l2_reg, lamda, p):
    ######### Build graph
    xavier_initializer = tf.contrib.layers.xavier_initializer()
    feature = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='inputs')

    #Input Layer
    input_layer = tf.reshape(feature, shape=[-1, 28, 28, 1])
    
    #3x3 convolutional layer, with 32 filters, using vertical and horizontal strides of 1.
    filter = tf.get_variable(name='filter', shape=[3, 3, 1, 32], initializer=xavier_initializer)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, 1, 1, 1], padding='SAME')#[batch, height, width, channels]
    b_0 = tf.get_variable(name='b_0', shape=[32], initializer=xavier_initializer)
    conv_layer += b_0
    
    #ReLU activation
    relu_1 = tf.nn.relu(conv_layer)
    
    #A batch normalization layer
    mean, variance = tf.nn.moments(conv_layer, axes=[0, 1, 2])
    bn_layer = tf.nn.batch_normalization(x=conv_layer, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-4)
    
    #A 2x2 max pooling layer
    max_pool = tf.nn.max_pool(value=bn_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    #Flatten layer
    flatten_layer = tf.reshape(max_pool, [-1, 6272])

    #Fully connected layer (with 784 output units, i.e. corresponding to each pixel)
    w_1 = tf.get_variable(name='w_1', shape=[6272, 784], initializer=xavier_initializer)
    b_1 = tf.get_variable(name='b_1', shape=[784], initializer=xavier_initializer)
    fc_layer1 = tf.matmul(flatten_layer, w_1) + b_1
    
    #2.3.2 dropout layer
    #ReLU activation
    prob = tf.placeholder(tf.float32)
    if drop_out:
        dropout_layer = tf.nn.dropout(fc_layer1, prob)
        relu_2 = tf.nn.relu(dropout_layer)
    else:
        relu_2 = tf.nn.relu(fc_layer1)

    #Fully connected layer (with 10 output units, i.e. corresponding to each class)
    w_2 = tf.get_variable(name='w_2', shape=[784,10], initializer=xavier_initializer)
    b_2 = tf.get_variable(name='b_2',shape=[10], initializer=xavier_initializer)
    fc_layer2 = tf.matmul(relu_2, w_2) + b_2

    #Softmax output
    label = tf.placeholder(dtype=tf.int32, shape=[None, 10], name="label")
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer2)
    
    #Cross Entropy loss
    pred = tf.argmax(input=fc_layer2, axis=1)
    CE_loss = tf.reduce_mean(entropy)

    #2.3.1 L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(scale=lamda)
    if l2_reg:
        reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        penalty = tf.contrib.layers.apply_regularization(regularizer, reg)
        CE_loss += penalty

    return CE_loss, pred, feature, label, prob

def accuracy(x_hat, x):
    count = 0
    for i in range(len(x)):
        if (x_hat[i] - x[i]).all():
            count += 1
    accur = count/len(x)
    return accur


def SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p):
    loss, pred, x, y, prob = NNTF(drop_put, l2_reg, lamda, p)
    adam_op = tf.train.AdamOptimizer(epsilon).minimize(loss)

    acc_train = np.zeros(epochs)
    loss_train = np.zeros(epochs)
    acc_valid = np.zeros(epochs)
    loss_valid = np.zeros(epochs)
    acc_test = np.zeros(epochs)
    loss_test = np.zeros(epochs)
    
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    
    trainBatches = int(N/batch_size)
    validBatches = int(V/batch_size)
    testBatches = int(T/batch_size)
    iterations = N // batch_size
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            loss_train = np.arange(N);
            np.random.shuffle(loss_train)
            loss_valid = np.arange(V);
            np.random.shuffle(loss_valid)
            loss_test = np.arange(T);
            np.random.shuffle(loss_test)
            
            trainData_label = trainData[loss_train]
            validData_label = validData[loss_valid]
            testData_label = testData[loss_test]
            
            trainTarget_label = trainTarget[loss_train]
            validTarget_label = validTarget[loss_valid]
            testTarget_label = testTarget[loss_test]
            
            for j in range(iterations):
                """data_shape = (batch_size, trainData.shape[1]*trainData.shape[2])
                target_shape = (batch_size, 1)"""
                batch_data = trainData[j*batch_size:(j+1)*batch_size, :]
                batch_target = trainTarget[j*batch_size:(j+1)*batch_size, :]
                _ = sess.run([adam_op], feed_dict={x: batch_data, y: batch_target, prob: p})
            
            train_loss_per_epoch, train_hat = sess.run([loss, pred], feed_dict={x: trainData_label, y: trainTarget_label, prob: p})
            valid_loss_per_epoch, valid_hat = sess.run([loss, pred], feed_dict={x: validData_label, y: validTarget_label, prob: p})
            test_loss_per_epoch, test_hat = sess.run([loss, pred], feed_dict={x: testData_label, y: testTarget_label, prob: p})
            
            loss_train[i] = train_loss_per_epoch
            loss_valid[i] = valid_loss_per_epoch
            loss_test[i] = test_loss_per_epoch

            acc_train[i] = accuracy(train_hat, trainTarget)
            acc_valid[i] = accuracy(valid_hat, validTarget)
            acc_test[i] = accuracy(test_hat, testTarget)

    plot_sgd_part2(loss_train, loss_valid, loss_test, acc_train, acc_valid, acc_test, lamda, p)
    print("loss_train = {loss}\n".forma(loss=loss_train[epochs-1]))
    print("loss_valid = {loss}\n".forma(loss=loss_valid[epochs-1]))
    print("loss_test = {loss}\n".forma(loss=loss_test[epochs-1]))
    print("acc_train = {acc}\n".forma(acc=acc_train[epochs-1]))
    print("acc_valid = {acc}\n".forma(acc=acc_valid[epochs-1]))
    print("acc_test = {acc}\n".forma(acc=acc_test[epochs-1]))
          

def plot_sgd_part2(loss_train, loss_valid, loss_test, acc_train, acc_valid, acc_test, lamda, p):
    ###############################################
    ###               2.2                       ###
    ###############################################
    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 4), sharey=False, facecolor='w')
        
    # loss plot
    ax0.set_ylabel("loss")
    ax0.set_xlabel("epochs")
    # accuracy plot
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("epochs")
        
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
    
    plt.savefig('SGD_part2_reg={lamda}_prob={prob}.png'.format(lamda=lamda, prob=p))
    plt.show()
    

#part 2.2
batch_size = 32
epochs = 50
epsilon = 1e-04
drop_out = False
l2_reg = False
lamda = 0.0
p = 0.0

SGD(batch_size, epochs, epsilon, drop_out, l2_reg, lamda, p)

#part 2.3.1
drop_out = False
l2_reg = True
lamda = 0.01
p = 0.0
SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p)

lamda = 0.1
SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p)

lamda = 0.05
SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p)

#part 2.3.2
drop_out = True
l2_reg = False
lamda = 0.0
p = 0.9
SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p)

p = 0.75
SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p)

p = 0.5
SGD(batch_size, epochs, epsilon, drop_put, l2_reg, lamda, p)
