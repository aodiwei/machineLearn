#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/2/7'
# 
"""
import text_classifier
from network2 import Network
textClassifier = text_classifier.TextClassifier(data_stream="VIVO")
training_data, test_data, X_vect, labels_count = textClassifier.make_data_for_network()
net = Network([X_vect, 30, labels_count])

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, epochs=1000, mini_batch_size=50, eta=0.01, evaluation_data=test_data, monitor_evaluation_accuracy=True)
# textClassifier.result_report(test_results)
print("evaluation_cost:{}\n evaluation_accuracy{}\n training_cost{}\n training_accuracy:{}".format(evaluation_cost, evaluation_accuracy, training_cost, training_accuracy ))