#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:06
# @Author: chenhr33733
# @File: dataset.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved

import torch
from torch.utils.data import Dataset
import pandas as pd
import re
from utility.utils import sentence_delimiters_pattern


t2v_ner_v2_labels = ['PRODUCT', 'LOC', 'EVENT', 'FAC', 'GPE', 'ORG', 'PERSON']


def generate_ner_labels(labels):
    label2id = dict()
    index = 0
    for label in labels:
        for prefix in ['B-', 'I-']:
            label2id[prefix + label] = index
            index += 1
    label2id['O'] = index
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def truncate_t2v_ner_v2(file_path):
    df = pd.read_json(file_path, lines=True)
    texts = []
    labels = []
    for index in range(len(df['text'])):
        sentence = df['text'][index]
        texts.append(sentence)
        label = ['O'][::] * len(sentence)
        for l in df['labels'][index]:
            start_id = l[0]
            end_id = l[1]
            label_name = l[2]
            tmp = ['B-'+label_name] + ['I-'+label_name][::]*(end_id-start_id-1)
            label[start_id:end_id] = tmp
        labels.append(label)

    truncate_texts = []
    truncate_labels = []
    for index in range(len(texts)):
        text = texts[index]
        label = labels[index]
        iteration = re.finditer(pattern=sentence_delimiters_pattern, string=text)
        indexes = [0] + [i.span()[1] for i in iteration] + [len(text)]
        t_texts = []
        t_labels = []
        for i in range(1, len(indexes)):
            x, y = indexes[i - 1], indexes[i]
            if text[x:y]:
                t_texts.append(text[x:y])
                t_labels.append(label[x:y])
        truncate_texts.extend(t_texts)
        truncate_labels.extend(t_labels)

    return truncate_texts, truncate_labels


class T2VNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id, _ = generate_ner_labels(t2v_ner_v2_labels)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], return_tensors='pt', padding='max_length', truncation=True, max_length=256)
        item = {key: val.squeeze() for key, val in item.items()}
        label = self.labels[idx][:256-2]
        label = ['O'] + label + ['O'][::]*(254-len(label)) + ['O']
        label = [self.label2id[i] for i in label]
        item['labels'] = torch.tensor(label).detach()
        return item

    def __len__(self):
        return len(self.labels)

    def sanity_check(self):
        raise NotImplementedError


if __name__ == '__main__':
    texts, labels = truncate_t2v_ner_v2('t2v_ner_v2_train.jsonl')
