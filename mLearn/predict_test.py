#!/usr/bin/env python
#coding:utf-8
"""
__title__ = ""
__author__ = "adw"
__mtime__ = "2016/6/23"
__purpose__ = 
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def generate_data3():
    X = [
        [514, 448],
        [516, 446],
        [539, 461],
        [522, 446],
        [569, 489],
        [543, 472],
    ]
    y = [485, 476, 501, 488, 537, 516]
    return np.array(X), np.array(y)

X, y = generate_data3()
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X, y)
pre_X = np.array([551, 457])
res = clf.predict(pre_X)
pass