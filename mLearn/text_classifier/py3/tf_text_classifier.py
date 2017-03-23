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
        self.mini_batch_size = 500
        self.mini_batches = None
        self.train_times = 100000
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.train_x = None
        self.train_y = None
        self.layers = Layers()
        self.make_data()

    def make_data(self):
        """

        :return:
        """
        textClassifier = text_classifier.TextClassifier(data_stream="MED")
        self.training_data, self.test_data, self.X_vect, self.labels_count = textClassifier.make_data_for_tensorflow()
        # self.x = tf.placeholder("float", shape=[None, X_vect])
        # self.y_ = tf.placeholder("float", shape=[None, labels_count])
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, self.X_vect], name='x_input')
            self.y_ = tf.placeholder(tf.float32, [None, self.labels_count], name='y_input')

        # n = len(self.training_data[0])
        # self.mini_batches = np.array(
        #         [(self.training_data[0][k: k + self.mini_batch_size], self.training_data[1][k: k + self.mini_batch_size]) for k in
        #          range(0, n, self.mini_batch_size)])
        self.train_x = self.training_data[0]
        self.train_y = self.training_data[1]
        self._num_examples = self.train_x.shape[0]

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.train_x = self.train_x[perm]
            self.train_y = self.train_y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.train_x[start:end], self.train_y[start:end]

    def train_with_tensorboard(self):
        """
        test Tensorboard
        :return:
        """
        activation_function = tf.nn.sigmoid
        l1 = self.layers.add_layer_with_tensorboard(self.x, self.X_vect, 30, layer_name="l1", activation_function=activation_function)
        l2 = self.layers.add_layer_with_tensorboard(l1, 30, 15, layer_name="l2", activation_function=activation_function)
        prediction = self.layers.add_layer_with_tensorboard(l2, 15, self.labels_count, layer_name="prediction", activation_function=tf.nn.softmax)

        with tf.name_scope("loss"):
            # loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - prediction), reduction_indices=[1]))
            cross_entropy = -tf.reduce_sum(self.y_ * tf.log(prediction))
            ce_sum = tf.scalar_summary("cross entropy", cross_entropy)

        with tf.name_scope("train"):
            train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss=cross_entropy)

        with tf.name_scope("test") as scope:
            correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)

        sess = tf.InteractiveSession()
        # 区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里
        # 接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir=logs/
        # 会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("logs/", sess.graph)
        # important step
        sess.run(tf.global_variables_initializer())

        # for i in range(self.train_times):
        #     for batch in self.mini_batches:
        #         result = sess.run([merged, accuracy], feed_dict={self.x: batch[0], self.y_: batch[1]})
        #     summary_str = result[0]
        #     acc = result[1]
        #     writer.add_summary(summary_str, i)
        #     print(i, acc)
        #     correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
        #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #     print(accuracy.eval(feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]}))

        batch = self.next_batch(self.mini_batch_size)
        for i in range(self.train_times):
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
            if i and i % 50 == 0:
                batch = self.next_batch(self.mini_batch_size)
                result = sess.run([merged, accuracy], feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]})
                summary_str = result[0]
                acc = result[1]
                print(i, acc)
                writer.add_summary(summary_str, i)
                # correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                # print(i, accuracy.eval(feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]}))

    def train(self):
        """

        :return:
        """
        dropout = tf.placeholder(tf.float32)
        l1 = self.layers.add_layer_with_drop(self.x, self.X_vect, 50, "layer1", activation_function=tf.nn.sigmoid)
        l2 = self.layers.add_layer_with_drop(l1, 50, 40, "layer2", activation_function=tf.nn.sigmoid)
        l3 = self.layers.add_layer_with_drop(l2, 40, 30, "layer3", activation_function=tf.nn.sigmoid)
        l4 = self.layers.add_layer_with_drop(l3, 30, 15, "layer4", activation_function=tf.nn.sigmoid)
        prediction = self.layers.add_layer_with_drop(l4, 15, self.labels_count, "layer_prediction", activation_function=tf.nn.softmax)

        saver = tf.train.Saver()

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        save_path = saver.save(sess, '../model_save/tf_model.ckpt')

        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(prediction))
        train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

        batch = self.next_batch(self.mini_batch_size)
        for i in range(self.train_times):
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
            if i and i % 50 == 0:
                batch = self.next_batch(self.mini_batch_size)
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                print(i, accuracy.eval(feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]}))

    def reload_model(self):
        """
        加载训练好的模型
        :return:
        """
        l1 = self.layers.add_layer_with_drop(self.x, self.X_vect, 50, "layer1", activation_function=tf.nn.sigmoid)
        l2 = self.layers.add_layer_with_drop(l1, 50, 40, "layer2", activation_function=tf.nn.sigmoid)
        l3 = self.layers.add_layer_with_drop(l2, 40, 30, "layer3", activation_function=tf.nn.sigmoid)
        l4 = self.layers.add_layer_with_drop(l3, 30, 15, "layer4", activation_function=tf.nn.sigmoid)
        prediction = self.layers.add_layer_with_drop(l4, 15, self.labels_count, "layer_prediction", activation_function=tf.nn.softmax)

        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        saver.restore(sess, '../model_save/tf_model.ckpt')

        # cross_entropy = -tf.reduce_sum(self.y_ * tf.log(prediction))
        # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy.eval(feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]}))

        # batch = self.next_batch(self.mini_batch_size)
        # for i in range(self.train_times):
        #     train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
        #     if i and i % 50 == 0:
        #         batch = self.next_batch(self.mini_batch_size)
        #         correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_, 1))
        #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #         print(i, accuracy.eval(feed_dict={self.x: self.test_data[0], self.y_: self.test_data[1]}))






if __name__ == "__main__":
    tfTextClassifier = TfTextClassifier()
    # tfTextClassifier.train_with_tensorboard()
    tfTextClassifier.train()
    # tfTextClassifier.reload_model()