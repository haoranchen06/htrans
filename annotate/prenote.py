#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:54
# @Author: chenhr33733
# @File: prenote.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import copy
import time
import jsonlines
import pandas as pd
import tqdm
import hanlp
import re
from utility.utils import *
from typing import List


class HanlpElectraForPrenote(object):
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.electra = hanlp.load('../pretrained_models/mtl/close_tok_pos_ner_srl_dep_sdp_con_electra_base_20210111_124519_20210304_140543')

    def hanlp_ner_labels_to_doccano_ner_labels(self, hanlp_res):
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

    def hanlp_ner_prenote(self, inputs, outputs_dir):
        """
        :param inputs: list of sentences
        :return:
        """
        line_template = {"text": "", "labels": []}
        labels_element_template = [0, 2, "ORG"]

        inputs = remove_empty_str(inputs)
        outputs = []
        electra = self.electra
        for sent in tqdm.tqdm(inputs):
            sent = text_preprocess(sent)
            line = copy.deepcopy(line_template)
            line['text'] = sent
            hanlp_res = electra(data=sent, tasks='ner/ontonotes')
            line['labels'] = self.hanlp_ner_labels_to_doccano_ner_labels(hanlp_res)
            outputs.append(line)

        with jsonlines.open(outputs_dir, 'w') as w:
            for sent in outputs:
                w.write(sent)


if __name__ == '__main__':
    df = pd.read_json(path_or_buf='data/nbd_data/20210317_finance_text.jl', lines=True)
    inputs = df['contents'][:100]
    prenote_model = HanlpElectraForPrenote()
    prenote_model.hanlp_ner_prenote(inputs=inputs, outputs_dir='data/prenote_sample.jsonl')