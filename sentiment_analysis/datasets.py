#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/7/19 9:48
# @Author: chenhr33733
# @File: datasets.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import torch
from torch.utils.data import Dataset
import pandas as pd


t2v_sa_v1_train_emos = {116: '振奋人心', 117: '幽默诙谐', 118: '温馨幸福', 119: '轻松惬意', 120: '紧张危机', 121: '伤感忧郁'}
t2v_sa_v1_train_pnn = {113: '正面', 114: '负面', 115: '中性'}
t2v_sa_v1_test_emos = {104: '振奋人心', 105: '幽默诙谐', 106: '温馨幸福', 107: '轻松惬意', 108: '紧张危机', 109: '伤感忧郁'}
t2v_sa_v1_test_pnn = {110: '中性', 111: '负面', 112: '正面'}

t2v_sa_v1_emos_refactor = {0: '振奋人心', 1: '幽默诙谐', 2: '温馨幸福', 3: '轻松惬意', 4: '紧张危机', 5: '伤感忧郁', 6: '其他'}
t2v_sa_v1_pnn_refactor = {0: '负面', 1: '中性', 2: '正面'}
label2emos = {value: key for key, value in t2v_sa_v1_emos_refactor.items()}
label2pnn = {value: key for key, value in t2v_sa_v1_pnn_refactor.items()}


def read_t2v_sa_v1_train_emos(file_path):
    df = pd.read_json(file_path, lines=True)
    texts = []
    labels = []
    for index in range(df.__len__()):
        cur_labels = {annotation['label'] for annotation in df['annotations'][index]}
        emos = cur_labels & t2v_sa_v1_train_emos.keys()
        pnn = cur_labels & t2v_sa_v1_train_pnn.keys()
        if emos:
            label = emos.pop()
            texts.append(df['text'][index])
            labels.append(label2emos[t2v_sa_v1_train_emos[label]])
        elif not emos and pnn:
            texts.append(df['text'][index])
            labels.append(label2emos['其他'])
    return texts, labels


def read_t2v_sa_v1_train_pnn(file_path):
    df = pd.read_json(file_path, lines=True)
    texts = []
    labels = []
    for index in range(df.__len__()):
        cur_labels = {annotation['label'] for annotation in df['annotations'][index]}
        pnn = cur_labels & t2v_sa_v1_train_pnn.keys()
        if pnn:
            label = pnn.pop()
            texts.append(df['text'][index])
            labels.append(label2pnn[t2v_sa_v1_train_pnn[label]])
    return texts, labels


def read_t2v_sa_v1_test_emos(file_path):
    df = pd.read_json(file_path, lines=True)
    texts = []
    labels = []
    for index in range(df.__len__()):
        cur_labels = {annotation['label'] for annotation in df['annotations'][index]}
        emos = cur_labels & t2v_sa_v1_test_emos.keys()
        pnn = cur_labels & t2v_sa_v1_test_pnn.keys()
        if emos:
            label = emos.pop()
            texts.append(df['text'][index])
            labels.append(label2emos[t2v_sa_v1_test_emos[label]])
        elif not emos and pnn:
            texts.append(df['text'][index])
            labels.append(label2emos['其他'])
    return texts, labels


def read_t2v_sa_v1_test_pnn(file_path):
    df = pd.read_json(file_path, lines=True)
    texts = []
    labels = []
    for index in range(df.__len__()):
        cur_labels = {annotation['label'] for annotation in df['annotations'][index]}
        pnn = cur_labels & t2v_sa_v1_test_pnn.keys()
        if pnn:
            label = pnn.pop()
            texts.append(df['text'][index])
            labels.append(label2pnn[t2v_sa_v1_test_pnn[label]])
    return texts, labels


class T2VSADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).detach() for key, val in self.encodings.items()}
        # item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).detach()
        return item

    def __len__(self):
        return len(self.labels)

    def sanity_check(self):
        raise NotImplementedError


if __name__ == '__main__':
    texts_res, labels_res = read_t2v_sa_v1_train_emos('t2v_sa_v1_train.jsonl')