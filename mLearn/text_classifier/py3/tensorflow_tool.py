#!/usr/bin/env python
# coding:utf-8

import tensorflow as tf

class tensorflowTool:

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        """
        添加一层
        :param inputs:
        :param in_size:
        :param out_size:
        :param activation_function:
        :return:
        """
        weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

        return outputs