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


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    nbd_list = ['20210317_finance_text.jl', '20210317_industry_text.jl', '20210319_money_text.jl', '20210319_stocks_text.jl']
    nbd_inputs = []
    tencent_inputs = []
    for i in nbd_list:
        df = pd.read_json(path_or_buf=os.path.join('data/nbd_data/', i), lines=True)
        inputs = df['contents'].tolist()[:1000]
        nbd_inputs += inputs
    for root, dirs, files in os.walk('data/tencent_data', topdown=False):
        for name in files:
            jl_dir = os.path.join(root, name)
            if jl_dir[-7:] == 'text.jl':
                df = pd.read_json(path_or_buf=jl_dir, lines=True)
                inputs = df['contents'].tolist()
                tencent_inputs += inputs
    tencent_inputs = simhash_slide(tencent_inputs)
    news_hybrid = nbd_inputs + tencent_inputs[:1000]
    result = simhash_slide(news_hybrid)
    random.shuffle(result)
    prenote_model = HanlpElectraForPrenote()
    prenote_model.hanlp_ner_prenote(inputs=result[:3000], outputs_dir='data/prenote_3000.jsonl')