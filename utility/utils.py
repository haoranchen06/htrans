#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:42
# @Author: chenhr33733
# @File: utils.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved
import compileall
import os
import re
import matplotlib.pyplot as plt
import torch

sentence_delimiters = {'?', '!', '？', '！', '。', '……', '…', '\n'}
sentence_delimiters_pattern = r'\?|!|？|！|。|……|…|\n'


def remove_empty_str(inputs):
    return [i for i in inputs if i]


def text_preprocess(sent):
    blank_pattern = u'( )|(\xa0)|(\u3000)|(\t)|(\n)|(▲)|(▼)'
    return re.sub(blank_pattern, '', sent)


def py2pyc(project_dir):
    compileall.compile_dir(project_dir)
    for root, dirs, files in os.walk(project_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            if name[-4:] == '.pyc':
                src = os.path.join(root, name)
                dst = re.sub(r'''.cpython-38''', '', src)
                dst = re.sub(r'''/__pycache__''', '', dst)
                os.rename(src, dst)
            elif name[-3:] == '.py':
                os.remove(os.path.join(root, name))


def compute_seq_classification_acc(x, y):
    """
    :param x: predicted labels
    :param y: validate labels
    :return: different labels' accuracy
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    labels = set(y.tolist())
    mask_dict = {label: y == label for label in labels}
    labels_acc = dict()
    matched_labels = x == y
    labels_acc[-1] = sum(matched_labels) / len(matched_labels)
    for i in labels:
        i_acc = sum(matched_labels * mask_dict[i]) / sum(mask_dict[i])
        labels_acc[i] = i_acc
    return labels_acc


def split_retain_pattern(pattern, text):
    iteration = re.finditer(pattern=pattern, string=text)
    indexes = [0] + [i.span()[1] for i in iteration] + [len(text)]
    sentences = []
    for i in range(1, len(indexes)):
        x, y = indexes[i-1], indexes[i]
        sentence = text[x:y]
        if sentence:
            sentences.append(sentence)
    return sentences


def make_plot(train_loss, val_loss, save_dir):
    plt.style.use('ggplot')
    plt.figure()
    plt.title('model')
    plt.xlabel('num of epoch')
    plt.ylabel('loss')
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train', 'validate'])
    plt.savefig(save_dir)
