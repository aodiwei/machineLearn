#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/1/24'
# 
"""
import tensorflow as tf
import numpy as np

# x_data = np.random.rand(100).astype('float32')
# y_data = x_data * 0.1 + 0.3
#
# W = tf.Variable(tf.random_uniform([1], -0.1, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = W * x_data + b
#
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     with tf.device("/cpu:1"):
#         sess.run(init)
#         for step in range(201):
#             sess.run(train)
#             if step % 20 == 0:
#                 print(step, sess.run(W), sess.run(b))
#

# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.mul(input1, intermed)
# with tf.Session() as sess:
#     result = sess.run(mul)
#     print(result)

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

ouput= tf.mul(input1, input2)

with tf.Session() as sess:
    result = sess.run([ouput], feed_dict={input1: [7.0, 3, 9], input2: [2.0, 33, 9]})
    print(result)