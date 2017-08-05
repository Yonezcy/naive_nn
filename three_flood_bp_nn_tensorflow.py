# -- encoding: utf-8 --
'在iris数据集上使用tensorflow搭建一个简单的3层神经网络'
__author__ = 'zcy'

import numpy as np
import tensorflow as tf
import random
from sklearn import datasets
from sklearn import preprocessing

iris = datasets.load_iris()

rand_sample = range(150)
random.shuffle(rand_sample)

# 训练集
min_max_scaler = preprocessing.MinMaxScaler()
train_data = iris.data[rand_sample[:100]]
train_data = min_max_scaler.fit_transform(train_data)

train_label = iris.target[rand_sample[:100]]
train_label = np.reshape(train_label, (100, 1))
train_label = min_max_scaler.fit_transform(train_label)

new_train_label = np.zeros([100, 3])
for i in range(100):
    if train_label[i] == 0: new_train_label[i] = [0, 0, 0]
    elif train_label[i] == 0.5: new_train_label[i] = [0, 1, 0]
    elif train_label[i] == 1: new_train_label[i] = [1, 0, 0]
train_label = new_train_label

# 测试集
test_data = iris.data[rand_sample[120:]]
test_data = min_max_scaler.fit_transform(test_data)

test_label = iris.target[rand_sample[120:]]
test_label = np.reshape(test_label, (30, 1))
test_label = min_max_scaler.fit_transform(test_label)

new_test_label = np.zeros([30, 3])
for i in range(30):
    if test_label[i] == 0: new_test_label[i] = [0, 0, 0]
    elif test_label[i] == 0.5: new_test_label[i] = [0, 1, 0]
    elif test_label[i] == 1: new_test_label[i] = [1, 0, 0]
test_label = new_test_label

x = tf.placeholder("float", shape=[None, 4])
y_ = tf.placeholder("float", shape=[None, 3])

W1 = tf.Variable(tf.truncated_normal([4, 3], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([3], 0.1))

layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([5, 3], stddev=0.1))
b2 = tf.Variable(tf.truncated_normal([3], 0.1))

y = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1000):
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: train_data, y_: train_label})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: train_data, y_: train_label})
    # 预测新样本类别
    # print sess.run(tf.argmax(y, 1), feed_dict={x: min_max_scaler.fit_transform([[2.0, 3.1, 1.3, 4.2]])})
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: test_data, y_: test_label}))
