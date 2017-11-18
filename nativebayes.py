#-*- coding:utf-8 –*-
import os
import re
import jieba
import jieba.analyse
import numpy as np
import logging
import random

def text_parse(document):
    word_list = jieba.cut(document)
    word_list = [word for word in word_list if word not in stopwords and not word.isdigit() and len(word) > 1]
    return word_list

def text_parse_tfidf(document):
    keyword_list = jieba.analyse.extract_tags(document)
    word_list = [word for word in keyword_list if word not in stopwords]
    return word_list

def load_train_data():
    doc_list = []
    cls_list = []
    file_list = os.listdir('data/train/C000007')
    for file_name in file_list:
        doc = open('data/train/C000007' + '/' + file_name,'r').read()
        doc_list.append(text_parse(doc))
        cls_list.append(7)
    file_list = os.listdir('data/train/C000024')
    for file_name in file_list:
        doc = open('data/train/C000024' + '/' + file_name,'r').read()
        doc_list.append(text_parse(doc))
        cls_list.append(24)
    return doc_list, cls_list

def load_test_data():
    doc_list = []
    cls_list = []
    file_list = os.listdir('data/test/C000007')
    for file_name in file_list:
        doc = open('data/test/C000007' + '/' + file_name,'r').read()
        doc_list.append(text_parse(doc))
        cls_list.append(7)
    file_list = os.listdir('data/test/C000024')
    for file_name in file_list:
        doc = open('data/test/C000024' + '/' + file_name,'r').read()
        doc_list.append(text_parse(doc))
        cls_list.append(24)
    return doc_list, cls_list

def create_vocab(documents):
    vocabulary = set([])
    for document in documents:
        vocabulary = vocabulary|set(document)
    return list(vocabulary)

def doc2vector(document, vocabulary):
    doc_vec = [0] * len(vocabulary)
    for word in document:
        if word in vocabulary:
            doc_vec[vocabulary.index(word)] = 1
        else:
            logging.info("word: %s is't in vocabulary"%word)
    return doc_vec
    
def load_stopword(stopword_file):
    stopwords = [line.strip() for line in open(stopword_file, 'r').readlines()]
    stopwords = [word.decode('utf-8') for word in stopwords]
    return set(stopwords)

def train(doc_list, cls_list):
    doc_num = len(doc_list)
    word_num = len(vocabulary)
    p_c07 = np.ones(word_num)
    denom_c07 = 2.0
    p_c24 = np.ones(word_num)
    denom_c24 = 2.0
    for i in range(doc_num):
        print 'process doc %d'%i
        doc_vector = doc2vector(doc_list[i], vocabulary)
        if cls_list[i] == 7:
            p_c07 = p_c07 + doc_vector
            denom_c07 = denom_c07 + sum(doc_vector)
        if cls_list[i] == 24:
            p_c24 = p_c24 + doc_vector
            denom_c25 = denom_c07 + sum(doc_vector)
    p_c07 = np.log(p_c07/denom_c07)
    p_c24 = np.log(p_c24/denom_c25)
    return p_c07, p_c24

def classify(doc_vec, p_c07, p_c24):
    p07 = sum(doc_vec * p_c07)
    p24 = sum(doc_vec * p_c24)

    if p07 > p24:
        return 7
    else:
        return 24

def test():
    doc_list, cls_list = load_test_data()
    testset = zip(doc_list, cls_list)
    random.shuffle(testset)
    doc_list, cls_list = zip(*testset)

    doc_num = len(doc_list)
    error_count = 0.0
    for i in range(doc_num):
        doc_vec = doc2vector(doc_list[i], vocabulary)
        label = classify(doc_vec, p_c07, p_c24)
        print "true label: %s, predict label: %s"%(cls_list[i], label)
        if label != cls_list[i]:
            error_count = error_count + 1
    print "error count %d, error rate %.2f%%"%(error_count, error_count*100/doc_num)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(filename)s] [line:%(lineno)d] [%(levelname)s] [%(message)s]', datefmt='%a, %d %b %Y %H:%M:%S', filename='myapp.log',filemode='w')
    stopwords = load_stopword('data/stop_words.txt')
    doc_list, cls_list = load_train_data()
    vocabulary = create_vocab(doc_list)
    p_c07, p_c24 = train(doc_list, cls_list)
    test()