#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 16:40
# @Author: chenhr33733
# @File: redup.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved
import os.path

from simhash import Simhash, SimhashIndex
import pandas as pd
import random
from tqdm import tqdm
import pickle
from utility.utils import text_preprocess, remove_empty_str


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
    nbd_list = ['20210317_finance_text.jl', '20210317_industry_text.jl', '20210319_money_text.jl', '20210319_stocks_text.jl']
    nbd_hybrid = []
    for i in nbd_list:
        df = pd.read_json(path_or_buf=os.path.join('data/nbd_data/', i), lines=True)
        inputs = df['contents'].tolist()[:1000]
        nbd_hybrid += inputs
    result = simhash_slide(nbd_hybrid)
