#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:54
# @Author: chenhr33733
# @File: prenote.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import copy
import random
import time
import jsonlines
import pandas as pd
import tqdm
import hanlp
import re
from utility.utils import *
from typing import List
from deduplicate import simhash_slide
import json
import requests
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from torch.utils.data import Dataset


class SeqForPredictDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class HanlpElectraForPrenote(object):
    def __init__(self):
        self.electra = hanlp.load('../pretrained_models/mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_base_20210111_124519_20210304_140543')

    def hanlp_ner_labels_to_doccano_ner_labels(self, hanlp_res):
        entity_types = {'ORG', 'PRODUCT', 'EVENT', 'PERSON', 'FAC', 'GPE', 'LOC'}
        tok = hanlp_res['tok/fine']
        ner = hanlp_res['ner/ontonotes']
        dp = [0][::] * (len(tok)+1)
        for id, t in enumerate(tok):
            dp[id+1] = dp[id] + len(t)
        labels = []
        for n in ner:
            if n[1] in entity_types:
                labels.append([dp[n[2]], dp[n[3]], n[1]])
        return labels

    def hanlp_ner_prenote(self, inputs, outputs_dir):
        """
        :param inputs: list of sentences
        :return:
        """
        line_template = {"text": "", "labels": []}
        labels_element_template = [0, 2, "ORG"]

        inputs = [text_preprocess(i) for i in inputs]
        inputs = remove_empty_str(inputs)
        outputs = []
        electra = self.electra
        for sent in tqdm.tqdm(inputs):
            line = copy.deepcopy(line_template)
            line['text'] = sent
            hanlp_res = electra(data=sent, tasks='ner/ontonotes')
            line['labels'] = self.hanlp_ner_labels_to_doccano_ner_labels(hanlp_res)
            outputs.append(line)

        with jsonlines.open(outputs_dir, 'w') as w:
            for sent in outputs:
                w.write(sent)


class T2VPrenote(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("../../text2video/models/chinese-roberta-wwm-ext")
        device = torch.device('cuda:2')
        self.emos_model = BertForSequenceClassification.from_pretrained("../../text2video/models/emos_weighted/checkpoint-600")
        self.emos_model.to(device)
        self.emos_model.eval()
        self.pnn_model = BertForSequenceClassification.from_pretrained("../../text2video/models/pnn_weighted/checkpoint-300")
        self.pnn_model.to(device)
        self.pnn_model.eval()
        self.device = device
        print('model initialize completed')

    def sa_prenote(self, inputs, outputs_dir):
        t2v_sa_v1_emos_refactor = {0: '振奋人心', 1: '幽默诙谐', 2: '温馨幸福', 3: '轻松惬意', 4: '紧张危机', 5: '伤感忧郁', 6: '其他'}
        t2v_sa_v1_pnn_refactor = {0: '负面', 1: '中性', 2: '正面'}
        line_template = {"text": "", "labels": []}
        padding_len = min(max([len(i) for i in inputs]), 128)
        tokens = self.tokenizer(inputs, truncation=True, padding=True, max_length=padding_len)
        dataset = SeqForPredictDataset(encodings=tokens)
        loader = DataLoader(dataset, batch_size=16)
        emos_prob = []
        pnn_prob = []
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                emos_outputs = self.emos_model(input_ids, attention_mask=attention_mask)
                emos_prob.append(torch.softmax(emos_outputs.logits, dim=1))
                pnn_outputs = self.pnn_model(input_ids, attention_mask=attention_mask)
                pnn_prob.append(torch.softmax(pnn_outputs.logits, dim=1))
        emos_prob = torch.cat(emos_prob, dim=0)
        pnn_prob = torch.cat(pnn_prob, dim=0)
        emos_labels = torch.argmax(emos_prob, dim=1).tolist()
        pnn_labels = torch.argmax(pnn_prob, dim=1).tolist()
        outputs = []
        for index in range(len(inputs)):
            line = copy.deepcopy(line_template)
            line['text'] = inputs[index]
            el = t2v_sa_v1_emos_refactor[emos_labels[index]]
            pl = t2v_sa_v1_pnn_refactor[pnn_labels[index]]
            line['labels'].extend([el, pl])
            outputs.append(line)

        with jsonlines.open(outputs_dir, 'w') as w:
            for sent in outputs:
                w.write(sent)


class sa_api(object):
    def __init__(self, host, sa_port):
        self.url = "http://" + host + ":" + sa_port + "/sa"

    def __call__(self, text):
        return self.forward(text=text)

    def forward(self, text):
        payload = {'text': text}
        response = requests.request("POST", self.url, data=payload)
        data = json.loads(response.text)
        body = data['body']
        return body


if __name__ == '__main__':
    with open('data/processed_data/t2v_sa_v2.txt') as f:
        inputs = f.read().split('\n')
    prenote_model = T2VPrenote()
    prenote_model.sa_prenote(inputs=inputs, outputs_dir='data/processed_data/t2v_sa_v2.jsonl')
