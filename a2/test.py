import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.maximum(x,0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
def computeLayer(X, W, b):
    return np.matmul(X,W)+b 

def CE(target, prediction):
    N=target.shape[0]
    return -np.sum(np.log(prediction)*target)/N

def gradCE(target, prediction):
    #https://deepnotes.io/softmax-crossentropy
    #-np.sum(target/prediction, axis=1)
    return prediction-target#

def gradOuterLayer(X_2, target, prediction):
    
    return np.matmul(np.transpose(X_2),(prediction-target))

def gradOuterBiases(X_2, target, prediction):
    N=target.shape[0]    
    biases = np.ones((N,1))
    return np.transpose(np.matmul(np.transpose(prediction-target),biases))


def reluD(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def gradHiddenLayer(X_1, S_1, Wo, target, prediction):
    S_d=reluD(S_1)
    Backgrad=np.matmul((prediction-target),np.transpose(Wo))*S_d
    
    return np.matmul(np.transpose(X_1),Backgrad)

def gradHiddenBiases(S_1, Wo, target, prediction):
    S_d=reluD(S_1)
    Backgrad=np.matmul((prediction-target),np.transpose(Wo))*S_d
    N=target.shape[0]    
    biases = np.ones((1,N))
    
    return np.matmul(biases,Backgrad)

def accuracy(Wo, bo, Wh, bh, target, data):
    layer1=computeLayer(data,Wh,bh)
    layer1_out=relu(layer1)
    layer2=computeLayer(layer1_out,Wo,bo)
    layer2_out=softmax(layer2)
    predict = np.argmax(layer2_out, axis = 1)
    real = np.argmax(target, axis = 1)
   
    return  np.sum(predict == real)/data.shape[0]


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

newtrain, newvalid, newtest=convertOneHot(trainTarget, validTarget, testTarget)
trainData=trainData.reshape(10000,784)
validData=validData.reshape(6000,784)
testData=testData.reshape(2724,784)
W_out=np.random.normal(0, np.sqrt(2.0/1010), (1000,10))#
b_out=np.zeros((1, 10))
W_hidden=np.random.normal(0, np.sqrt(2.0/(1000+784)), (784,1000))#
b_hidden=np.zeros((1, 1000))
V_out=np.full((1000, 10), 1e-5)
V_hidden=np.full((784, 1000), 1e-5)

i=0
epochs=200
learning_rate=0.0000001
Vold_out=V_out
Vnew_out=V_out
Vbold_out=b_out
Vbnew_out=b_out

Vold_hidden=V_hidden
Vnew_hidden=V_hidden
Vbold_hidden=b_hidden
Vbnew_hidden=b_hidden

accuracy_set = []
valid_accuracy_set = []
test_accuracy_set = []
loss_set = []
valid_loss_set = []
test_loss_set = []


for i in range(200):  

    v_layer1_in=computeLayer(validData,W_hidden,b_hidden)
    v_layer1_out=relu(v_layer1_in)
    v_layer2_in=computeLayer(v_layer1_out,W_out,b_out)
    v_layer2_out=softmax(v_layer2_in)
    v_pred_data = np.argmax(v_layer2_out, axis = 1)
    v_real_data = np.argmax(newvalid, axis = 1)
    
    N=validData.shape[0]
    v_y=np.sum(v_pred_data == v_real_data)/N
    valid_accuracy_set.append(v_y)
    valid_loss_set.append(CE(newvalid,v_layer2_out))


    t_layer1_in=computeLayer(testData,W_hidden,b_hidden)
    t_layer1_out=relu(t_layer1_in)
    t_layer2_in=computeLayer(t_layer1_out,W_out,b_out)
    t_layer2_out=softmax(t_layer2_in)
    t_pred_data = np.argmax(t_layer2_out, axis = 1)
    t_real_data = np.argmax(newtest, axis = 1)
    
    N=testData.shape[0]
    t_y=np.sum(t_pred_data == t_real_data)/N
    test_accuracy_set.append(t_y)
    test_loss_set.append(CE(newtest,t_layer2_out))

    layer1_in=computeLayer(trainData,W_hidden,b_hidden)
    layer1_out=relu(layer1_in)
    layer2_in=computeLayer(layer1_out,W_out,b_out)
    layer2_out=softmax(layer2_in)
    pred_data = np.argmax(layer2_out, axis = 1)
    real_data = np.argmax(newtrain, axis = 1)
    
    N=trainData.shape[0]
    y=np.sum(pred_data == real_data)/N
    accuracy_set.append(y)
    loss_set.append(CE(newtrain,layer2_out))
    print(y)
    # outer layer
    Vnew_out=0.99*Vold_out+learning_rate*gradOuterLayer(layer1_out, newtrain, layer2_out)
    W_out=W_out-Vnew_out
    Vold_out=Vnew_out
    
    Vbnew_out=0.99*Vbold_out+learning_rate*gradOuterBiases(layer1_out, newtrain, layer2_out)
    b_out=b_out-Vbnew_out    
    Vbold_out=Vbnew_out
    
    Vnew_hidden=0.99*Vold_hidden+learning_rate*gradHiddenLayer(trainData, layer1_in, W_out, newtrain, layer2_out)
    W_hidden=W_hidden-Vnew_hidden
    Vold_hidden=Vnew_hidden    
    
    Vbnew_hidden=0.99*Vbold_hidden+learning_rate*gradHiddenBiases(layer1_in, W_out, newtrain, layer2_out)
    b_hidden=b_hidden-Vbnew_hidden
    Vbold_hidden=Vbnew_hidden    



plt.figure(1)
plt.title("train data accuracy: ephocs=200 learning rate=0.0001")
plt.plot(accuracy_set)
plt.plot(valid_accuracy_set)
plt.plot(test_accuracy_set)
pic_name = "accuracy_set.png"
plt.xlabel("epochs")
plt.ylabel("accuracy_set")
plt.legend(['train', 'valid','test'], loc='lower right')
#
#plt.figure(2)
#plt.title("CE loss set: ephocs=200 learning rate=0.0001")
#plt.plot(loss_set)
#plt.plot(valid_loss_set)
#plt.plot(test_loss_set)
#pic_name = "loss_set.png"
#plt.xlabel("epochs")
#plt.ylabel("loss_set")
#plt.legend(['train', 'valid','test'], loc='upper right')
#
#plt.figure(3)
#plt.title("test data accuracy: ephocs=200 learning rate=0.0001 hidden units=100")
#plt.plot(test_accuracy_set)
#pic_name = "test_set.png"
#plt.xlabel("epochs")
#plt.ylabel("accuracy_set")
##plt.legend(['100', '500','2000'], loc='lower right')
#
#
#plt.figure(4)
#plt.title("test data accuracy: ephocs=200 learning rate=0.0001")
#plt.plot(test_loss_set)
#pic_name = "test_ce_set.png"
#plt.xlabel("epochs")
#plt.ylabel("loss_set")
#plt.legend(['100', '500','2000'], loc='upper right')
#
#plt.show()
