#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:06
# @Author: chenhr33733
# @File: dataset.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved
import copy
import sys
sys.path.append('..')

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from utility.utils import sentence_delimiters_pattern
from transformers import BertTokenizer
from ner_metrics import SeqEntityScore


t2v_ner_v2_labels = ['PRODUCT', 'LOC', 'EVENT', 'FAC', 'GPE', 'ORG', 'PERSON']


def generate_ner_labels(labels):
    label2id = dict()
    label2id['O'] = 0
    index = 1
    for label in labels:
        for prefix in ['B-', 'I-']:
            label2id[prefix + label] = index
            index += 1
    # label2id['O'] = index
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


# def truncate_t2v_ner_v2(file_path):
#     df = pd.read_json(file_path, lines=True)
#     texts = []
#     labels = []
#     for index in range(len(df['text'])):
#         sentence = df['text'][index]
#         texts.append(sentence)
#         label = ['O'][::] * len(sentence)
#         for l in df['labels'][index]:
#             start_id = l[0]
#             end_id = l[1]
#             label_name = l[2]
#             tmp = ['B-'+label_name] + ['I-'+label_name][::]*(end_id-start_id-1)
#             label[start_id:end_id] = tmp
#         labels.append(label)
#
#     truncate_texts = []
#     truncate_labels = []
#     for index in range(len(texts)):
#         text = texts[index]
#         label = labels[index]
#         iteration = re.finditer(pattern=sentence_delimiters_pattern, string=text)
#         indexes = [0] + [i.span()[1] for i in iteration] + [len(text)]
#         t_texts = []
#         t_labels = []
#         for i in range(1, len(indexes)):
#             x, y = indexes[i - 1], indexes[i]
#             if text[x:y]:
#                 t_texts.append(text[x:y])
#                 t_labels.append(label[x:y])
#         truncate_texts.extend(t_texts)
#         truncate_labels.extend(t_labels)
#
#     return truncate_texts, truncate_labels


def truncate_t2v_ner_v2(file_path):
    """
    :param file_path: t2v_ner_v2 dataset path
    :return: truncate_texts and corresponding label_list

    assume that label_list is sorted with the index order
    """
    df = pd.read_json(file_path, lines=True)
    tc_texts, tc_labels = [], []
    for ts, ls in zip(df['text'], df['labels']):
        iteration = re.finditer(pattern=sentence_delimiters_pattern, string=ts)
        indexes = [0] + [i.span()[1] for i in iteration] + [len(ts)]
        lb_id = 0
        icm = 0
        # increment of ts length
        for i in range(1, len(indexes)):
            x, y = indexes[i - 1], indexes[i]
            if x == y:
                continue
            tc_texts.append(ts[x:y])
            sub_lb_list = []
            while lb_id < len(ls):
                if x <= ls[lb_id][0] <= y and x <= ls[lb_id][1] <= y:
                    lb = ls[lb_id][::]
                    lb[0] -= icm
                    lb[1] -= icm
                    sub_lb_list.append(lb)
                    lb_id += 1
                elif ls[lb_id][0] > y:
                    break
                else:
                    lb_id += 1
            tc_labels.append(sub_lb_list)
            icm += y-x
    return tc_texts, tc_labels


class T2VNERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        assert isinstance(tokenizer, transformers.models.bert.tokenization_bert.BertTokenizer)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id, self.id2label = generate_ner_labels(t2v_ner_v2_labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        lb_list = self.labels[idx]
        item = self.convert(text, lb_list)
        item = self.pad_and_truncate(item, max_len=256)
        for key in item.keys():
            item[key] = torch.tensor(item[key]).detach()
        return item

    def __len__(self):
        return len(self.labels)

    def convert(self, text, lb_list):
        tokenizer = self.tokenizer
        prev_id = 0
        last_id = len(text)
        elements = []
        for lb in lb_list:
            start_id = lb[0]
            end_id = lb[1]
            non_label = [prev_id, start_id, 'non_label']
            elements.append(non_label)
            elements.append(lb)
            prev_id = end_id
        non_label = [prev_id, last_id, 'non_label']
        elements.append(non_label)

        tokens, new_lb_list = [], []
        sum_len = 0
        for i in elements:
            token = tokenizer.tokenize(text[i[0]:i[1]])
            tokens += token
            tok_len = len(token)
            if i[2] != 'non_label':
                new_lb = i[::]
                new_lb[0] = sum_len
                new_lb[1] = sum_len + tok_len
                new_lb_list.append(new_lb)
            sum_len += tok_len

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0][::] * len(input_ids)
        attention_mask = [1][::] * len(input_ids)
        label_ids = [0][::] * len(input_ids)
        for new_lb in new_lb_list:
            start_id = new_lb[0]
            end_id = new_lb[1]
            label_name = new_lb[2]
            tmp = ['B-'+label_name] + ['I-'+label_name][::]*(end_id-start_id-1)
            label_ids[start_id:end_id] = [self.label2id[i] for i in tmp]

        item = dict()
        item['input_ids'] = input_ids
        item['token_type_ids'] = token_type_ids
        item['attention_mask'] = attention_mask
        item['labels'] = label_ids
        return item

    def pad_and_truncate(self, item, max_len=256):
        tokenizer = self.tokenizer
        item = copy.deepcopy(item)
        cls = tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = tokenizer.convert_tokens_to_ids(['[SEP]'])
        for key in item.keys():
            item[key] = item[key][:max_len-2]
        cur_len = len(item['input_ids'])
        item['input_ids'] = cls + item['input_ids'] + sep + [0][::]*(max_len-2-cur_len)
        item['token_type_ids'] = [0] + item['token_type_ids'] + [0] + [0][::]*(max_len-2-cur_len)
        item['attention_mask'] = [1] + item['attention_mask'] + [1] + [0][::]*(max_len-2-cur_len)
        item['labels'] = [0] + item['labels'] + [0] + [0][::]*(max_len-2-cur_len)
        return item

    def sanity_check(self):
        raise NotImplementedError


if __name__ == '__main__':
    texts, labels = truncate_t2v_ner_v2('t2v_ner_v2_train.jsonl')
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    dataset = T2VNERDataset(texts=texts, labels=labels, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=2)
    metrics = SeqEntityScore(id2label=dataset.id2label)
