#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ""
__author__ = "adw"
__mtime__ = "2016/6/8"
__purpose__ = 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()



''''' 数据读入 '''
data = []
labels = []
with open("data.txt") as ifile:
    for line in ifile:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)

''''' 标签转换为0/1 '''
y[labels == 'fat'] = 1

''''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

''''' 使用信息熵作为划分标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)

''''' 把决策树结构写入文件 '''
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

'''''测试结果的打印'''
answer = clf.predict(x_train)
print(x_train)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

plot_decision_boundary(clf.predict, x_train, y_train)

'''''准确率与召回率'''
precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict_proba(x)[:, 1]
print(classification_report(y, answer, target_names=['thin', 'fat']))
