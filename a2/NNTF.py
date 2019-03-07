import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from starter import loadData, convertOneHot
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget) # Nx10


def NN():
    ######### Build graph
    initializer = tf.contrib.layers.xavier_initializer()
    W_0 = tf.Variable(initializer([K, 10]))
    b_0 = tf.Variable(tf.zeros([1, 10]))

