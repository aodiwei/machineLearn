#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/2/7'
# 
"""
import tensorflow as tf
import numpy as np
import text_classifier

textClassifier = text_classifier.TextClassifier(data_stream="VIVO")
training_data, test_data, X_vect, labels_count = textClassifier.make_data_for_tensorflow()

x = tf.placeholder("float", shape=[None, X_vect])
y_ = tf.placeholder("float", shape=[None, labels_count])

W = tf.Variable(tf.zeros(shape=[X_vect, labels_count]))
b = tf.Variable(tf.zeros(shape=[labels_count]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

n = len(training_data[0])
mini_batch_size = 10
mini_batches = np.array([(training_data[0][k: k + mini_batch_size], training_data[1][k: k + mini_batch_size]) for k in range(0, n, mini_batch_size)])

# for i in range(1000):
#     batch = (training_data[0][i:i*50], training_data[1][i:i*50])
#     # sess.run(train_step)
#     ret = train_step.run(feed_dict={x: batch[0], y_: batch[1]})

for batch in mini_batches:
    ret = train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy)
print(accuracy.eval(feed_dict={x: test_data[0], y_: test_data[1]}))