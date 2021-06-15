#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:09
# @Author: chenhr33733
# @File: utils.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import copy
import os
import re
import compileall
import time

import jsonlines
import pandas as pd
import tqdm
import hanlp
import re


def remove_empty_str(inputs):
    return [i for i in inputs if i]


def text_preprocess(sent):
    return re.sub(' ', '', sent)


def hanlp_style_to_doccano_style(hanlp_res):
    entity_types = {'ORG', 'NORP', 'PRODUCT', 'EVENT', 'PERSON', 'FAC', 'GPE', 'LOCATION'}
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


if __name__ == '__main__':
    line_template = {"text": "", "labels": []}
    labels_element_template = [0, 2, "ORG"]

    dir = '../data/nbd_data/finance_text.jl'
    df = pd.read_json(path_or_buf=dir, lines=True)
    file1 = df['contents'][:100].tolist()
    file1 = remove_empty_str(file1)

    file2 = []

    # entity_types = {'ORG', 'NORP', 'PRODUCT', 'EVENT', 'PERSON', 'FAC', 'GPE', 'LOCATION'}
    electra = hanlp.load('../models/pretrained_models/mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_base_20210111_124519_20210304_140543')
    start = time.time()
    # electra(data=file1, tasks='ner')
    for i in tqdm.tqdm(file1):
        i = text_preprocess(i)
        line = copy.deepcopy(line_template)
        line['text'] = i
        hanlp_res = electra(data=i, tasks='ner/ontonotes')
        line['labels'] = hanlp_style_to_doccano_style(hanlp_res)
        file2.append(line)
    end = time.time()
    print(end-start)

    with jsonlines.open('file2.jsonl', 'w') as w:
        for i in file2:
            w.write(i)


