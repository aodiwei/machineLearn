#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ""
__author__ = "adw"
__mtime__ = "2016/6/28"
__purpose__ = 
"""

import pandas as pd

import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


# sns.set(style="ticks", color_codes=True)

# read csv file directly from a URL and save the results
# data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# # visualize the relationship between the features and the response using scatterplots
# # fig = sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8)
#
# g = sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.8, kind='reg')
# # g.savefig()
# # create a python list of feature names
# feature_cols = ['TV', 'Radio', 'Newspaper']
#
# # use the list to select a subset of the original DataFrame
# X = data[feature_cols]
# # select a Series from the DataFrame
# y = data['Sales']
#
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# from sklearn.linear_model import LinearRegression
#
# linreg = LinearRegression()
#
# linreg.fit(X_train, y_train)
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# y_pred = linreg.predict(X_test)
# print np.sqrt(metrics.mean_squared_error(y_test, y_pred))

def linera_predict(data, predict_data=None, plot=False):
    """

    :param data: pandas dataFrame
    :param plot: is plot
    :return:
    """
    cols = data._series.keys()
    cols = ["wsnum", "total", "lev1", "lev2", "target"]
    cols = ["lev1", "lev2", "target"]
    feature_cols = cols[:-1]
    y_cols = cols[-1]
    X = data[feature_cols]
    y = data[y_cols]
    if plot:
        # sns.plt.figure()
        # sns.pairplot(data, x_vars=feature_cols, y_vars=y_cols, size=7, aspect=0.8)
        sns.pairplot(data, x_vars=feature_cols, y_vars=y_cols, size=7, aspect=0.8, kind='reg')
        sns.plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    y_pred = linreg.predict(X_test)
    print linreg.score(X_test, y_test)
    error_rate = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print u"训练数据误差：", error_rate
    if predict_data:
        pred_result = linreg.predict(predict_data)
        print u"预测结果：", pred_result


if __name__ == "__main__":
    # data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
    data = pd.read_csv("gkdata.csv")
    # pre = [[230.1, 37.8, 69.2]]
    pre = [[96939, 324678, 551, 475]]
    pre = [[522, 446]]
    linera_predict(data, plot=True, predict_data=pre)
