import os
import time
import math
import numpy as np
import tensorflow as tf
import cv2
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Library for plot the output and save to file
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)

# Pre processing data
# Normalize the data by subtract the mean image
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage

# Select device
deviceType = "/gpu:0"

# Your Own Model
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32)

def yourOwnModel():
    with tf.device(deviceType):
        #---------------------C1---------------------------
        wConv1 = tf.get_variable("wConv1", shape=[5, 5, 3, 32])
        bConv1 = tf.get_variable("bConv1", shape=[32])
        c1 = tf.nn.conv2d(x, wConv1, strides=[1, 1, 1, 1],padding="SAME") + bConv1

        #N1  Normalization(LRU)
        norm1 = tf.nn.local_response_normalization(c1)
        #Relu
        p1 = tf.nn.relu(norm1)
        #P1
        r_p1 = tf.nn.max_pool(p1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#output size = 32/2 = 16

        #---------------------C2----------------------------
        wConv2 = tf.get_variable("wConv2", shape=[5, 5, 32, 64])
        bConv2 = tf.get_variable("bConv2", shape=[64])
        c2 = tf.nn.conv2d(r_p1, wConv2, strides=[1, 1, 1, 1], padding="SAME") + bConv2

        #N2
        norm2 = tf.nn.local_response_normalization(c2)
        #Relu
        p2 = tf.nn.relu(norm2)
        #P2
        r_p2 = tf.nn.max_pool(p2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")#output size is 16/2 = 8

        #---------------------C3--------------------------------
        wConv3 = tf.get_variable("wConv3", shape=[5, 5, 64, 128])
        bConv3 = tf.get_variable("bConv3", shape=[128])
        c3 = tf.nn.conv2d(r_p2, wConv3, strides=[1, 1, 1, 1],padding="SAME") + bConv3

        #N3
        norm3 = tf.nn.local_response_normalization(c3)
        #Relu
        p3 = tf.nn.relu(norm3)
        #P3
        r_p3 = tf.nn.max_pool(p3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #output size is 8/2 = 4

        #-------------------C4----------------------------
        wConv4 = tf.get_variable("wConv4", shape=[5, 5, 128, 100])
        bConv4 = tf.get_variable("bConv4", shape=[100])
        c4 = tf.nn.conv2d(r_p3, wConv4, strides=[1, 1, 1, 1],padding="SAME") + bConv4

        #N4
        norm4 = tf.nn.local_response_normalization(c4)
        #Relu
        p4 = tf.nn.relu(norm4)
        #P4
        r_p4 = tf.nn.max_pool(p4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #output size is 4/2 = 2

        #-----------------F1-------------------------------
        h3 = tf.reshape(r_p4,[-1,400]) #2*2*100
        w4 = tf.get_variable("w4",shape=[400,10])
        b4 = tf.get_variable("b4",shape=[10])

        y_pre = tf.matmul(h3,w4) + b4

        #drop out
        yOut = tf.nn.dropout(y_pre, keep_prob)

        # Define Loss
        # softmax loss function
        totalLoss = tf.losses.softmax_cross_entropy(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)

        # Define Optimizer
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

        # Define correct Prediction and accuracy
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

        return [meanLoss, accuracy, trainStep, yOut]


def train(Model, xT, yT, xV, yV, xTe, yTe, batchSize, epochs=100, printEvery=10):
    # Train Model
    trainIndex = np.arange(xTrain.shape[0])
    np.random.shuffle(trainIndex)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # Mini-batch
            losses = []
            accs = []
            # For each batch in training data
            for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                # Get the batch data for training
                startIndex = (i * batchSize) % xTrain.shape[0]
                idX = trainIndex[startIndex:startIndex + batchSize]
                currentBatchSize = yTrain[idX].shape[0]

                # Train
                loss, acc, _ = sess.run(Model[0:3], feed_dict={x: xT[idX, :], y: yT[idX],keep_prob: 0.5})

                # Collect all mini-batch loss and accuracy
                losses.append(loss * currentBatchSize)
                accs.append(acc * currentBatchSize)

            totalAcc = np.sum(accs) / float(xTrain.shape[0])
            totalLoss = np.sum(losses) / xTrain.shape[0]
            if e % printEvery == 0:
                print("{0:.0f}/10       {1:.3f}       {2:.2f}%".format((e + 10) / 10, totalLoss, totalAcc * 100), end='')
                loss, acc = sess.run(Model[0:2], feed_dict={x: xTe, y: yTe, keep_prob: 1})
                print("       {0:.3f}       {1:.2f}%".format(loss, acc * 100))

        # loss, acc = sess.run(Model[0:2], feed_dict={x: xTe, y: yTe , keep_prob: 1})
        # print('Testing loss = {0:.3f} and testing accuracy = {1:.2f}%'.format(loss, acc * 100))
        saver = tf.train.Saver()
        saver_path = saver.save(sess, "./model/model.ckpt")
        print("Model saved in file: ", saver_path)

        pass

# Start training  model
# print("Loop    Train Loss    Train Acc %   Test Loss   Test Acc %")
# train(yourOwnModel(), xTrain, yTrain, xVal, yVal, xTest, yTest,100)

def test(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (32, 32))
    # print(img.shape)

    img = img[np.newaxis,:]
    # print(img.shape)

    ximgTest = img.astype(np.float)
    yimgTest = np.ones(1,)
    # print(ximgTest.shape)
    # print(yimgTest.shape)

    # define predictioin
    yOut = yourOwnModel()[3]
    predict_op = tf.argmax(yOut, 1)

    #define 1st Conv Layer
    with tf.variable_scope('', reuse=True):
        wConv1 = tf.get_variable("wConv1")
        bConv1 = tf.get_variable("bConv1")
        c1 = tf.nn.conv2d(x, wConv1, strides=[1, 1, 1, 1], padding="SAME") + bConv1

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./model/model.ckpt")
        print("load model success")

        #predict img
        predict_result = sess.run(predict_op, feed_dict={x: ximgTest, y: yimgTest, keep_prob: 1})
        # print(predict_result)
        print("The prediction:",classesName[predict_result[0]])

        #save img
        layer_img = sess.run(c1, feed_dict={x: ximgTest, y: yimgTest, keep_prob: 1})
        for i in range (32):
            save_img = layer_img[:, :, :, i]
            save_img.shape = [32,32]
            plt.subplot(6,6,i+1)
            plt.imshow(save_img, cmap='gray')
            plt.axis('off')
            plt.imsave('CONV_rslt.png',save_img)
        plt.savefig(baseDir + 'CONV_rslt.png')
        plt.clf()
        print("save visualization result as CONV_rslt.png")

    cv2.waitKey(5)

    pass

# test('./model/plane.png')
# test('./model/dog.png')
# test('./model/car.png')


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        print("Loop    Train Loss    Train Acc %   Test Loss   Test Acc %")
        train(yourOwnModel(), xTrain, yTrain, xVal, yVal, xTest, yTest,100)
    elif sys.argv[1] == 'test' and len(sys.argv)>2:
        for i in range(2,len(sys.argv)):
            file = sys.argv[i]
            # print(file)
            test (file)
    else:
        print('Command error')
