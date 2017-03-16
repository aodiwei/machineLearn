#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/3/16'
# 
"""
import tensorflow as tf
import numpy as np
import text_classifier
from layers import Layers


class TfTextClassifier:
    def __init__(self):
        self.x = None
        self.y_ = None
        self.training_data = None
        self.test_data = None
        self.X_vect = None
        self.labels_count = None
        self.mini_batch_size = 50
        self.mini_batches = None
        self.train_times = 1000
        self.layers = Layers()
        self.make_data()

    def make_data(self):
        """

        :return:
        """
        textClassifier = text_classifier.TextClassifier(data_stream="VIVO")
        self.training_data, self.test_data, self.X_vect, self.labels_count = textClassifier.make_data_for_tensorflow()
        # self.x = tf.placeholder("float", shape=[None, X_vect])
        # self.y_ = tf.placeholder("float", shape=[None, labels_count])
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.X_vect], name='x_input')
            self.y_ = tf.placeholder(tf.float32, [None, self.labels_count], name='y_input')

        n = len(self.training_data[0])
        self.mini_batches = np.array(
                [(self.training_data[0][k: k + self.mini_batch_size], self.training_data[1][k: k + self.mini_batch_size]) for k in
                 range(0, n, self.mini_batch_size)])

    def train_with_tensorboard(self):
        """
        test Tensorboard
        :return:
        """
        l1 = self.layers.add_layer_with_tensorboard(self.x, self.X_vect, 30, activation_function=tf.nn.sigmoid)
        l2 = self.layers.add_layer_with_tensorboard(l1, 30, 15, activation_function=tf.nn.sigmoid)
        prediction = self.layers.add_layer_with_tensorboard(l2, 15, self.labels_count, activation_function=tf.nn.softmax)

        with tf.name_scope("loss"):
            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - prediction), reduction_indices=[1]))
            cross_entropy = -tf.reduce_sum(self.y_ * tf.log(prediction))

        with tf.name_scope("train"):
            train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss=cross_entropy)

        sess = tf.InteractiveSession()
        # 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里
        # 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
        # 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
        writer = tf.train.SummaryWriter("board_logs/", sess.graph)
        # important step
        sess.run(tf.global_variables_initializer())

        for i in range(self.train_times):
            for batch in self.mini_batches:
                train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(accuracy.eval(feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]}))

    def test(self):
        # define placeholder for inputs to network
        # 区别：大框架，里面有 inputs x，y
        with tf.name_scope('inputs'):
            xs = tf.placeholder(tf.float32, [None, self.X_vect], name='x_input')
            ys = tf.placeholder(tf.float32, [None, self.labels_count], name='y_input')

        # add hidden layer
        l1 = self.layers.add_layer_with_tensorboard(xs, self.X_vect, 30, activation_function=tf.nn.sigmoid)
        # add output layer
        prediction = self.layers.add_layer_with_tensorboard(l1, 30, self.labels_count, activation_function=tf.nn.softmax)

        # the error between prediciton and real data
        # 区别：定义框架 loss
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                                reduction_indices=[1]))

        # 区别：定义框架 train
        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        sess = tf.Session()

        # 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里
        # 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
        # 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
        writer = tf.train.SummaryWriter("logs/", sess.graph)
        # important step
        sess.run(tf.initialize_all_variables())


if __name__ == "__main__":
    tfTextClassifier = TfTextClassifier()
    # tfTextClassifier.train_with_tensorboard()
    tfTextClassifier.test()
