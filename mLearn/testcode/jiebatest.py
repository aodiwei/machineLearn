#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2016/9/10'
# 
"""
import jieba
import jieba.analyse

# string = u"唇裂术后外形不佳16年,唇裂术后继发畸形左侧"
string = "【10-02 #vivo柔光自拍X7#】这个手机用这个系统有问题 [用相册点多张分享到微信时出现没有响应严重发热怎么回事],"

jieba.analyse.set_stop_words('stopword.txt')
with open('stopword.txt', 'r') as f:
    stopwords = set([w.strip() for w in f])
comma_tokenizer = lambda x: jieba.cut(x, cut_all=False)

res = jieba.cut(string, cut_all=True)
for x in res:
    print x
print "="*80
res = jieba.cut(string, cut_all=False)
for x in res:
    print x
print "="*80
jieba.analyse.set_stop_words('stopword.txt')
tags = jieba.analyse.extract_tags(string, topK=20, withWeight=True)
for x in tags:
    print x[0], x[1]
