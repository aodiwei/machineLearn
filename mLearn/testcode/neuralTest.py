#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ""
__author__ = "adw"
__mtime__ = "2016/6/18"
__purpose__ = 
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons


def generate_data():
    np.random.seed(0)
    X, y = make_moons(100, noise=0.50)
    return X, y

def generate_data2():
    X = [
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1]
    ]
    y = [1, 1, 1, 1, 0, 1, 0]

    return np.array(X), np.array(y)

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


def show_original_data(X, y):
    # X, y = make_moons(200, noise=0.2)
    # # print X, y
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

#
# # 训练逻辑回归分类器
# clf = LogisticRegressionCV()
# clf.fit(X, y)
# plot_decision_boundary(lambda x: clf.predict(x), X, y)
# plt.title("Logistic Regression")


# ********三层神经网络*************
class Config:
    # num_examples = len(X)  # 训练集规模
    nn_input_dim = 2  # 输入层维度
    nn_output_dim = 1  # 输出层维度

    # 精心挑选的梯度下降参数
    epsilon = 0.01  # 梯度下降的学习速率
    reg_lambda = 0.01  # 规范化强度


# 实现之前定义的损失函数，这将用来评估我们的模型。

def calculate_loss(model, X, y):
    num_examples = len(X)  # training set size
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


#  用于预测输出的辅助函数
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# 将学习神经网络的参数， 并返回模型
# nn_hdim: 隐藏层中的结点
# num_passes: 用于梯度下降训练数据的数量
# print_loss 如果返回True， 每一千次迭代打印一次损失
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    num_examples = len(X)
    # 用随机值初始化参数
    np.random.seed(0)
    W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.nn_output_dim))

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += Config.reg_lambda * W2
        dW1 += Config.reg_lambda * W1

        # Gradient descent parameter update
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" % (i, calculate_loss(model, X, y)))
            # print "W1: {W1}, b1: {b1}, W2: {W2}, b2: {b2}".format(**model)

    return model


def visualize(X, y, model):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Logistic Regression")
    # plt.show()


X, y = generate_data3()
show_original_data(X, y)
# print X, y
model = build_model(X, y, 4,  print_loss=True)
yy = []
for x in X:
    res = predict(model, x)
    yy.append(res)
yy = np.array(yy)
# plt.contourf(X[:, 0],  X[:, 1], 7, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=yy, cmap=plt.cm.Spectral)
plt.show()
visualize(X, y, model)