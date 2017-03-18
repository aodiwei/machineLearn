#!/usr/bin/env python
# coding:utf-8

import tensorflow as tf


class Layers:
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

    def add_layer_with_drop(self, inputs, in_size, out_size, layer_name, activation_function=None, keep_prob=0.60):
        """
        添加一层
        :param layer_name:
        :param keep_prob: 保持多少不被drop
        :param inputs:
        :param in_size:
        :param out_size:
        :param activation_function:
        :return:
        """
        weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        wx_plus_b = tf.matmul(inputs, weights) + biases
        # 在 Wx_plus_b 上drop掉一定比例
        # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
        wx_plus_b = tf.nn.dropout(wx_plus_b, keep_prob)

        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)

        tf.histogram_summary(layer_name + '/outputs', outputs)

        return outputs

    def add_layer_with_tensorboard(self, inputs, in_size, out_size, layer_name, activation_function=None):
        """
        添加一层， 有tensorboard
        :param layer_name:
        :param inputs:
        :param in_size:
        :param out_size:
        :param activation_function:
        :return:
        """
        with tf.name_scope('layer' + layer_name):
            with tf.name_scope('weights' + layer_name):
                weights = tf.Variable(tf.random_normal([in_size, out_size]), name="weights" + layer_name)
            with tf.name_scope('biases' + layer_name):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name="biases" + layer_name)
            with tf.name_scope('wx_plus_b' + layer_name):
                wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)

            w_hist = tf.histogram_summary("weights" + layer_name, weights)
            b_hist = tf.histogram_summary("biases" + layer_name, biases)

            if activation_function is None:
                outputs = wx_plus_b
            else:
                outputs = activation_function(wx_plus_b)

            y_hist = tf.histogram_summary("y" + layer_name, outputs)

            return outputs
