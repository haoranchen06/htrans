#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/7/19 13:16
# @Author: chenhr33733
# @File: t2v_sa_v1.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


from datasets import *
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AdamW
import os
from tqdm import tqdm
from utility.utils import compute_seq_classification_acc
from sklearn.model_selection import train_test_split


def train_pnn():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 3
    train_texts, train_labels = read_t2v_sa_v1_train_pnn('t2v_sa_v1_train.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=0, stratify=train_labels)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    train_dataset = T2VSADataset(encodings=train_encodings, labels=train_labels)
    val_dataset = T2VSADataset(encodings=val_encodings, labels=val_labels)

    training_args = TrainingArguments(
        output_dir='./results/pnn_baseline',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs/pnn_baseline',  # directory for storing logs
        logging_steps=10,
    )

    model = BertForSequenceClassification.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext",
                                                          config=bert_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()
    trainer.evaluate()


def train_emos():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 7
    train_texts, train_labels = read_t2v_sa_v1_train_emos('t2v_sa_v1_train.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2, random_state=0, stratify=train_labels)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    train_dataset = T2VSADataset(encodings=train_encodings, labels=train_labels)
    val_dataset = T2VSADataset(encodings=val_encodings, labels=val_labels)

    training_args = TrainingArguments(
        output_dir='./results/emos_baseline',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=50,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs/emos_baseline',  # directory for storing logs
        logging_steps=10,
    )

    model = BertForSequenceClassification.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext",
                                                          config=bert_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()


def test_emos():
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    # bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    # bert_config.num_labels = 3
    test_texts, test_labels = read_t2v_sa_v1_train_emos('t2v_sa_v1_train.jsonl')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = T2VSADataset(encodings=test_encodings, labels=test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda:1')
    model = BertForSequenceClassification.from_pretrained("results/emos_baseline/checkpoint-2500")
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
            # res.append(outputs.cpu())
    acc = compute_seq_classification_acc(predict, test_labels)
    return acc


def test_pnn():
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    # bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    # bert_config.num_labels = 3
    test_texts, test_labels = read_t2v_sa_v1_train_pnn('t2v_sa_v1_train.jsonl')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    test_dataset = T2VSADataset(encodings=test_encodings, labels=test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda:1')
    model = BertForSequenceClassification.from_pretrained("results/pnn_baseline/checkpoint-2500")
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
            # res.append(outputs.cpu())
    acc = compute_seq_classification_acc(predict, test_labels)
    return acc


if __name__ == '__main__':
    res = test_emos()
