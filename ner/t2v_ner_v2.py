#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/8/6 15:51
# @Author: chenhr33733
# @File: t2v_ner_v2.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved

import sys
sys.path.append('..')

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
from models import BertCrfForNer
from ner_metrics import SeqEntityScore
import logging
from callback.lr_scheduler import get_linear_schedule_with_warmup
from collections import OrderedDict
from utility.utils import make_plot
from pprint import pprint


def train_ner_diy():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 15
    train_texts, train_labels = truncate_t2v_ner_v2('t2v_ner_v2_train.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,
                                                                        random_state=0)
    train_dataset = T2VNERDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer)
    val_dataset = T2VNERDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128)
    val_loader = DataLoader(val_dataset, batch_size=128)
    model = BertCrfForNer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext", config=bert_config)

    logging.info('Commencing training!')
    torch.manual_seed(196)

    device = torch.device('cuda')
    model.to(device)

    # freezing = False
    # if freezing:
    #     for param in model.base_model.parameters():
    #         param.requires_grad = False

    num_epochs = 4
    t_total = len(train_loader) * num_epochs

    weight_decay = 0.01
    crf_learning_rate = 1e-3
    learning_rate = 3e-5
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    lstm_param_optimizer = list(model.lstm.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate},

        {'params': [p for n, p in lstm_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': crf_learning_rate},
        {'params': [p for n, p in lstm_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': crf_learning_rate}
    ]

    optim = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.02 * t_total),
                                                num_training_steps=t_total)
    train_loss, val_loss = [], []

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    for epoch in range(num_epochs):
        train_loss_acc = 0
        model.train()
        progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=True, disable=False)

        for i, batch in enumerate(progress_bar):
            optim.zero_grad()
            batch = tuple(t.cuda() for t in batch.values())
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "labels": batch[3]}
            outputs = model(**inputs, return_dict=True)
            loss = outputs[0]
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            loss.backward()
            optim.step()
            scheduler.step()

            tr_loss, batch_size = loss.item(), len(inputs['labels'])
            stats = OrderedDict()
            stats['tr_loss'] = tr_loss
            train_loss_acc += tr_loss
            stats['learning_rate'] = optim.param_groups[0]['lr']
            stats['batch_size'] = batch_size
            progress_bar.set_postfix({key: '{:.4g}'.format(value) for key, value in stats.items()},
                                     refresh=True)

        train_loss.append(train_loss_acc / len(progress_bar))
        val_loss.append(validate_ner(model, val_loader))

        # model.cpu()
        os.makedirs("results/ner_crf_lr", exist_ok=True)
        model_to_save = (model.module if hasattr(model, "module") else model)
        torch.save(model_to_save.state_dict(), "results/ner_crf_lr/model_epoch{}.pt".format(epoch))
        # model.to(device)

    make_plot(train_loss, val_loss, save_dir="results/ner_crf_lr/loss.png")


def validate_ner(model, val_loader):
    model.eval()
    val_loss_acc = 0
    progress_bar = tqdm(val_loader, desc='| Validate', leave=True, disable=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            batch = tuple(t.cuda() for t in batch.values())
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "labels": batch[3]}
            outputs = model(**inputs, return_dict=True)
            loss = outputs[0]
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            val_loss, batch_size = loss.item(), len(inputs['labels'])
            stats = OrderedDict()
            stats['val_loss'] = val_loss
            val_loss_acc += val_loss
            stats['batch_size'] = batch_size
            progress_bar.set_postfix({key: '{:.4g}'.format(value) for key, value in stats.items()},
                                     refresh=True)

    return val_loss_acc / len(progress_bar)


def train_ner():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config = BertConfig.from_json_file("../pretrained_models/chinese-roberta-wwm-ext/config.json")
    bert_config.num_labels = 15
    train_texts, train_labels = truncate_t2v_ner_v2('t2v_ner_v2_train.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,
                                                                        random_state=0)
    train_dataset = T2VNERDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer)
    val_dataset = T2VNERDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir='./results/ner_crf',  # output directory
        num_train_epochs=4,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs/ner_crf',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy='epoch',
        save_total_limit=10,
        seed=42,
        # gradient_accumulation_steps=4,
        save_strategy='epoch',
    )

    model = BertCrfForNer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext", config=bert_config)

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset  # evaluation dataset
    )

    trainer.train()


def test_ner():
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    train_texts, train_labels = truncate_t2v_ner_v2('t2v_ner_v2_train.jsonl')
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2,
                                                                        random_state=0)
    val_dataset = T2VNERDataset(texts=val_texts, labels=val_labels, tokenizer=tokenizer)
    test_loader = DataLoader(val_dataset, batch_size=128)
    id2label = val_dataset.id2label

    device = torch.device('cuda:0')
    # model = BertCrfForNer.from_pretrained("results/ner_baseline/checkpoint-2056")
    bert_config = BertConfig.from_pretrained("../pretrained_models/chinese-roberta-wwm-ext")
    bert_config.num_labels = 15
    model = BertCrfForNer(config=bert_config)
    model.load_state_dict(torch.load("results/ner_crf_lr/model_epoch3.pt"))

    model.to(device)
    model.eval()
    predict_labels, gold_labels = [], []
    metrics = SeqEntityScore(id2label=id2label)
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = tuple(t.cuda() for t in batch.values())
            inputs = {"input_ids": batch[0], "token_type_ids": batch[1], "attention_mask": batch[2], "labels": batch[3]}
            outputs = model.forward(**inputs, return_dict=True)
            tags = model.crf.decode(outputs.logits, mask=inputs['attention_mask']).squeeze().cpu()
            if len(tags.shape) == 1:
                tags = tags.unsqueeze(0)
            pre_lb = tags.tolist()
            # pre_lb = torch.argmax(outputs.logits, dim=2).cpu().tolist()

            for i, j, k in zip(pre_lb, inputs['labels'].tolist(), inputs['attention_mask'].tolist()):
                mask_cnt = sum(k)
                predict_labels.append(i[1:mask_cnt-1])
                gold_labels.append(j[1:mask_cnt-1])

    metrics.update(gold_labels, predict_labels)
    return metrics.result()


def predict_ner():
    raise NotImplementedError


if __name__ == '__main__':
    train_ner_diy()
    # test_res = test_ner()
    # pprint(test_res, sort_dicts=True)
    pass
