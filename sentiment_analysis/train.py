#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/7/19 13:16
# @Author: chenhr33733
# @File: t2v_sa_v1.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved
import torch

from datasets import *
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from torch.nn.functional import normalize
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import os
from tqdm import tqdm
from utility.utils import compute_seq_classification_acc
from sklearn.model_selection import train_test_split
from models import BertForSCWithWeight


v1_emos_train_texts, v1_emos_train_labels = read_t2v_sa_v1_train_emos('t2v_sa_v1_train.jsonl')
v2_emos_train_texts, v2_emos_train_labels = read_t2v_sa_v2_train_emos('t2v_sa_v2.jsonl')
emos_train_texts = v1_emos_train_texts + v2_emos_train_texts
emos_train_labels = v1_emos_train_labels + v2_emos_train_labels
emos_test_texts, emos_test_labels = read_t2v_sa_v1_test_emos('t2v_sa_v1_test.jsonl')

v1_pnn_train_texts, v1_pnn_train_labels = read_t2v_sa_v1_train_pnn('t2v_sa_v1_train.jsonl')
v2_pnn_train_texts, v2_pnn_train_labels = read_t2v_sa_v2_train_pnn('t2v_sa_v2.jsonl')
pnn_train_texts = v1_pnn_train_texts + v2_pnn_train_texts
pnn_train_labels = v1_pnn_train_labels + v2_pnn_train_labels
pnn_test_texts, pnn_test_labels = read_t2v_sa_v1_test_pnn('t2v_sa_v1_test.jsonl')


def train_pnn():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 3
    train_texts, train_labels = pnn_train_texts, pnn_train_labels
    # train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2,
    #                                                                     random_state=0, stratify=train_labels)
    val_texts, val_labels = pnn_test_texts, pnn_test_labels
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    train_dataset = T2VSADataset(encodings=train_encodings, labels=train_labels, require_weight=False)
    val_dataset = T2VSADataset(encodings=val_encodings, labels=val_labels, require_weight=False)

    training_args = TrainingArguments(
        output_dir='./results/pnn_baseline',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs/pnn_baseline',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        save_total_limit=10,
        seed=42,
        save_strategy='epoch',
    )

    model = BertForSequenceClassification.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext", config=bert_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()


def train_emos():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 7
    train_texts, train_labels = emos_train_texts, emos_train_labels
    # train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,
    #                                                                     random_state=0, stratify=train_labels)
    val_texts, val_labels = emos_test_texts, emos_test_labels
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    train_dataset = T2VSADataset(encodings=train_encodings, labels=train_labels, require_weight=True)
    val_dataset = T2VSADataset(encodings=val_encodings, labels=val_labels, require_weight=True)

    training_args = TrainingArguments(
        output_dir='./results/emos_ghm',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs/emos_ghm',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        save_total_limit=10,
        seed=42,
        save_strategy='epoch',
    )

    model = BertForSCWithWeight.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext", config=bert_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()


def test_emos():
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    test_texts, test_labels = emos_test_texts, emos_test_labels
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = T2VSADataset(encodings=test_encodings, labels=test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained("results/emos_ghm/checkpoint-624")
    model.to(device)
    model.eval()
    predict = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predict.extend(torch.argmax(outputs.logits, dim=1).tolist())
    acc = compute_seq_classification_acc(predict, test_labels)
    return acc


def test_pnn():
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    test_texts, test_labels = pnn_test_texts, pnn_test_labels
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = T2VSADataset(encodings=test_encodings, labels=test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda:1')
    model = BertForSequenceClassification.from_pretrained("results/pnn_baseline/checkpoint-600")
    model.to(device)
    model.eval()
    predict = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predict.extend(torch.argmax(outputs.logits, dim=1).tolist())
    acc = compute_seq_classification_acc(predict, test_labels)
    return acc


if __name__ == '__main__':
    # train_emos()
    res = test_emos()
