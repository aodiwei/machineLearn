#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'asus'
__mtime__ = '2017/2/1'
__purpose__ = 
"""
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(" training data: {}".format(training_data[0]))
with open("log.log", "wb") as f:
    f.write(" training data: {}".format(training_data[0]))
net = network.Network([784, 8, 10])
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)