#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 16:40
# @Author: chenhr33733
# @File: redup.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved

import sys
sys.path.append('..')

import os.path
import re
from simhash import Simhash, SimhashIndex
import pandas as pd
import random
from tqdm import tqdm
import pickle
from utility.utils import *
import jsonlines


def simhash_slide(inputs):
    """
    :param inputs: list of sentences
    :return:
    """
    result = []
    inputs = [text_preprocess(i) for i in inputs]
    inputs = remove_empty_str(inputs)
    index = SimhashIndex(objs=[], k=10)
    for idx, v in tqdm(enumerate(inputs)):
        s = Simhash(v)
        dups = index.get_near_dups(s)
        if not dups:
            index.add(str(idx), s)
            result.append(v)

    return result


def simhash_tfidf():
    raise NotImplementedError


if __name__ == '__main__':
    random.seed(42)
    df = pd.read_json(path_or_buf='data/tencent_data/20210802_all_text.jl', lines=True)
    inputs = df['contents'].tolist()
    result = simhash_slide(inputs)
    random.shuffle(result)

    sentences = []
    for i in result:
        sentences.extend(split_retain_pattern(pattern=sentence_delimiters_pattern, text=i))
    sentences = [i for i in sentences if len(i) >= 3]
    with open('data/t2v_sa_v2.txt', 'w') as f:
        f.write('\n'.join(sentences))
