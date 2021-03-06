#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/1/21'
# 
"""
import csv
import logging
import time

import jieba

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize

CSV_PATH = r"../data/features_20170323_175026.csv"
# CSV_PATH = r"../data/bbgdata2.csv"
STOPWORD_PATH = "../config/stopword.txt"
FEATURES = 1000
VECTORIZER = "TFIDF"  # HASH
VIVO_CLEARN_DATA = r"../data/sample0.csv"
DATA_STREAM = "MED"  # MED/VIVO

train_name = time.time()
logging.basicConfig(filename="../logs/text_classifier_{}.log".format(train_name), level=logging.INFO,
                    format='%(asctime)s %(message)s')

log = logging.getLogger()


# lines = np.loadtxt("error.csv", delimiter=',', dtype='str', skiprows=0)

class TextClassifier:
    def __init__(self, csv_path=CSV_PATH, stopword_path=STOPWORD_PATH, features=FEATURES, vect=VECTORIZER, data_stream=DATA_STREAM):
        self.csv_path = csv_path
        self.stopword_path = stopword_path
        self.features = features
        self.vect = vect
        self.data_stream = data_stream

    def read_data_from_csv_de(self, path):
        """
        从csv文件里读取数据
        :param path:
        :return:
        """
        lines = np.loadtxt(path, delimiter=',', dtype='str', skiprows=1)
        return lines

    def read_data_from_csv(self, path):
        """
        从csv文件里读取数据
        :param path:
        :return:
        """
        lines = []
        with open(path, "rt", encoding="utf-8") as f:
            spamreader = csv.reader(f)
            for line in spamreader:
                lines.append(line)

        return lines

    def clean_vivo_data(self, path):
        """

        :param path:
        :return:
        """
        lines = []
        with open(path, 'rb') as f:
            ids = []
            spamreader = csv.reader(f)
            count = 0
            with open(VIVO_CLEARN_DATA, "wb") as fd:
                csv_writer = csv.writer(fd)
                for line in spamreader:
                    count += 1
                    if len(line) != 5 or "t" not in line[0] or line[0] in ids:
                        # print (line)
                        continue
                    # if line[0] not in ids:
                    # lines.append(line)
                    ids.append(line[0])
                    # if len(ids) > 1000:
                    # break
                    print(line)
                    csv_writer.writerow(line)
        # with open("bbg_clean_data.txt", "w") as f:
        # json.dump(lines, f)
        return lines

    def get_data(self, path, single_label_count=180):
        """
        获取每个标签等量的数据
        :param single_label_count:
        :param path:
        :return:
        """
        labels = {}
        data_items = []
        with open(path, "rt", encoding="utf-8") as f:
            first = 0
            spamreader = csv.reader(f)
            for line in spamreader:
                # 忽略第一行title
                if first is 0:
                    first = False
                    continue
                labels[line[-1]] = labels[line[-1]] + 1 if line[-1] in labels else 1
                # if labels[line[-1]] > single_label_count:
                #     continue
                # else:
                data_items.append(line)
        # log.info("total data count: {}".format(len(data_items)))
        # log.info("total data label count: {}".format(len(labels)))
        # log.info("total data dict: {}".format(labels))
        # print("data dict: {}".format(labels))

        # 过滤掉不满count数的记录
        # data_items_filiter = [x for x in data_items if labels[x[4]] >= count]
        data_items_filiter = filter(lambda x: labels[x[-1]] >= single_label_count, data_items)
        data_items_filiter_list = list(data_items_filiter)
        log.info("train data count: {}".format(len(data_items_filiter_list)))
        return data_items_filiter_list

    def word_tokenizer(self, word):
        """
        分词器
        :return:
        """
        return jieba.cut(word, cut_all=True)

    def stopwords(self):
        """
        停词
        :param path:
        :return:
        """
        with open(self.stopword_path, 'r', encoding="utf-8") as f:
            stopwords = set([w.strip() for w in f])

        return stopwords

    def vectorize(self, words):
        """
        转化文本为矩阵
        :param words:
        :return:
        """
        if self.vect == "TFIDF":
            vect_data = self.tfidf_vectorize(words)
        elif self.vect == "HASH":
            vect_data = self.hash_vectorize(words)
        else:
            raise TypeError("nonsupport")

        return vect_data

    def hash_vectorize(self, words):
        """
        转化文本为矩阵
        :param words:
        :return:
        """
        stopwords = self.stopwords()
        v = HashingVectorizer(tokenizer=self.word_tokenizer, stop_words=stopwords, n_features=self.features,
                              non_negative=True)
        words_data = v.fit_transform(words).toarray()
        return words_data

    def tfidf_vectorize(self, words):
        """
        转化文本为矩阵
        :param words:
        :return:
        """
        stopwords = self.stopwords()
        tfidf_v = TfidfVectorizer(tokenizer=self.word_tokenizer, stop_words=stopwords)
        tfidf_words_data = tfidf_v.fit_transform(words).toarray()
        vocabulary = tfidf_v.vocabulary_
        joblib.dump(vocabulary, '../model_save/vocabulary.pkl')

        return tfidf_words_data

    def make_data_for_network(self, limit=None):
        """
        为神经网络训练make数据
        :param limit:
        :return:
        """
        X_train, X_test, y_train, y_test = self.make_data(limit=limit)
        X_vect = len(X_train[0])
        labels_set = list(set(y_train))
        labels_count = len(labels_set)
        train_labels_vect = label_binarize(y_train, classes=labels_set)
        test_labels_vect = label_binarize(y_test, classes=labels_set)

        training_inputs = [np.reshape(x, (X_vect, 1)) for x in X_train]
        training_results = self.foramt_labels(train_labels_vect)
        training_data = list(zip(training_inputs, training_results))

        test_inputs = [np.reshape(x, (X_vect, 1)) for x in X_test]
        test_labels_vect = self.digitization_labels(test_labels_vect)
        test_data = list(zip(test_inputs, test_labels_vect))

        return training_data, test_data, X_vect, labels_count

    def make_data_for_tensorflow(self, limit=None):
        """

        :param limit:
        :return:
        """
        X_train, X_test, y_train, y_test = self.make_data(limit=limit)
        X_vect = len(X_train[0])
        labels_set = list(set(y_train))
        labels_count = len(labels_set)
        train_labels_vect = label_binarize(y_train, classes=labels_set)
        test_labels_vect = label_binarize(y_test, classes=labels_set)

        # training_inputs = [np.reshape(x, (X_vect, 1)) for x in X_train]

        # test_inputs = [np.reshape(x, (X_vect, 1)) for x in X_test]

        training_data = (X_train, train_labels_vect)
        test_data = (X_test, test_labels_vect)

        return training_data, test_data, X_vect, labels_count

    def foramt_labels(self, labels_vect):
        """
        为神经网络训练格式化标签
        :param labels_vect:
        :return:
        """
        labels = []
        for i in labels_vect:
            label = []
            for j in i:
                label.append([j])
            labels.append(label)

        return labels

    def digitization_labels(self, labels_vect):
        """
        把标签格式为为数字
        :param labels_vect:
        :return:
        """
        labels = []
        for i in labels_vect:
            for index, j in enumerate(i):
                if j == 1:
                    labels.append(index)
                    break

        return labels

    def make_data(self, train_index=1, label_index=-1, limit=None):
        """
        生成数据
        :param limit:
        :param label_index:
        :param train_index:
        :return:
        """
        X_data = []
        y_data = []
        if self.data_stream == "MED":
            data_lines = self.get_data(self.csv_path, single_label_count=40)
        elif self.data_stream == "VIVO":
            data_lines = self.get_data(VIVO_CLEARN_DATA, single_label_count=500)
        else:
            raise TypeError("nonsupport stream")
        if limit and limit < len(data_lines):
            data_lines = data_lines[:limit]

        for line in data_lines:
            if isinstance(train_index, (list, tuple)):
                train_text = ""
                for index in train_index:
                    train_text += line[index]
                X_data.append(train_text)
            else:
                X_data.append(line[train_index])
            y_data.append(line[label_index])
        self.cache = X_data

        X_data = self.vectorize(X_data)
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(X_data), np.asarray(y_data), test_size=0.25)
        return X_train, X_test, y_train, y_test

    def svm_text_classifier(self, train_index=1, label_index=-1):
        """
        svm 分类
        :param label_index:
        :param train_index:
        :return:
        """
        X_train, X_test, y_train, y_test = self.make_data(train_index=train_index, label_index=label_index)
        param_grid = {
            "C": [1e2, 5e2, 1e3, 5e3, 1e4, 5e5, 1e5, 1e6],
            "gamma": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        }
        # clf = GridSearchCV(SVC(kernel="rbf"), param_grid=param_grid)
        if self.data_stream == "MED":  # medical

            clf = SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape=None, degree=3, gamma=0.1, kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        elif self.data_stream == "VIVO":  # VIVO
            # clf = SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3,
            #           gamma=0.0001, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
            #           tol=0.001, verbose=False)
            # clf = SVC(C=5000.0, cache_size=200, class_weight=None, coef0=0.0,
            #           decision_function_shape=None, degree=3, gamma=0.0001, kernel='rbf',
            #           max_iter=-1, probability=False, random_state=None, shrinking=True,
            #           tol=0.001, verbose=False)
            clf = SVC(C=1000.0, cache_size=200, class_weight=None, coef0=0.0,
                      decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
                      max_iter=-1, probability=False, random_state=None, shrinking=True,
                      tol=0.001, verbose=False)
        else:
            raise TypeError("nonsupport stream")

        clf = clf.fit(X_train, y_train)

        # log.info("best_estimator_:{}".format(clf.best_estimator_))
        # log.info("best_params_:{}".format(clf.best_params_))
        # log.info("best_score_:{}".format(clf.best_score_))

        joblib.dump(clf, '../model_save/svm_model.pkl')
        y_pred = clf.predict(X_test)
        self.show_result(y_test, y_pred)

    def show_result(self, y_test, y_pred):
        """
        展示结果
        :param y_test:
        :param y_pred:
        :return:
        """
        labels_right = {}
        labels_error = {}
        for index, y in enumerate(y_test):
            if y == y_pred[index]:
                labels_right[y] = labels_right[y] + 1 if y in labels_right else 1
            else:
                labels_error[y] = labels_error[y] + 1 if y in labels_error else 1

        right_total = 0
        for k, v in labels_right.items():
            total = v + labels_error[k] if k in labels_error else v
            log.info("{}:{}".format(k, (1.0 * v) / total))
            right_total += v

        log.info("total precision:{}".format((1.0 * right_total) / len(y_test)))

        log.info("=" * 20 + "svm classifier" + "=" * 20)
        log.info(classification_report(y_test, y_pred))

        # log.info("finish")
        log.info(confusion_matrix(y_test, y_pred))

    def pred_new_text(self, text):
        """
        分类新的文本
        :param text:
        :return:
        """
        clf = joblib.load('../model_save/svm_model.pkl')
        vocabulary = joblib.load('../model_save/vocabulary.pkl')
        tfidf_v = TfidfVectorizer(tokenizer=self.word_tokenizer, stop_words=self.stopwords(), vocabulary=vocabulary)
        x = tfidf_v.fit_transform([text]).toarray()
        y = clf.predict(x)

        return y

    def bayes_text_classifier(self, train_index=1, label_index=-1):
        """
        bayes
        :param label_index:
        :param train_index:
        :return:
        """
        X_train, X_test, y_train, y_test = self.make_data(train_index=train_index, label_index=label_index)
        clf = MultinomialNB(alpha=0.01)
        # train
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        labels_right = {}
        labels_error = {}
        for index, y in enumerate(y_test):
            if y == y_pred[index]:
                labels_right[y] = labels_right[y] + 1 if y in labels_right else 1
            else:
                labels_error[y] = labels_error[y] + 1 if y in labels_error else 1

        right_total = 0
        for k, v in labels_right.iteritems():
            total = v + labels_error[k] if k in labels_error else v
            log.info("{}:{}".format(k, (1.0 * v) / total))
            right_total += v

        log.info("total precision:{}".format((1.0 * right_total) / len(y_test)))

        msg = "=" * 20 + "bayes classifier" + "=" * 20
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

    def result_report(self, result):
        """

        :param result:
        :return:
        """
        y_test = []
        y_pred = []
        for re in result:
            y_pred.append(re[0])
            y_test.append(re[1])
        log.info(classification_report(y_test, y_pred))
        log.info(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    textClassifier = TextClassifier()
    textClassifier.svm_text_classifier()
    y = textClassifier.pred_new_text("自觉双侧面下部宽大，影响美观3年余")
    print(y)
    pass
