#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/8/6 15:51
# @Author: chenhr33733
# @File: t2v_ner_v2.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved

import torch
from datasets import *
from torch.utils.data import DataLoader
from collections import defaultdict, Counter
from torch.nn.functional import normalize
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, Trainer, TrainingArguments, AdamW
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from models import BertForSCWithWeight


def train_ner():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 15
    train_texts, train_labels = truncate_t2v_ner_v2('t2v_ner_v2_train.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,
                                                                        random_state=0)
    train_dataset = T2VNERDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer)
    val_dataset = T2VNERDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./results/ner_baseline',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs/ner_baseline',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        save_total_limit=10,
        seed=42,
        gradient_accumulation_steps=4,
        save_strategy='epoch',
    )

    model = BertForTokenClassification.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext", config=bert_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()


if __name__ == '__main__':
    train_ner()
