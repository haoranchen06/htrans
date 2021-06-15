#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 16:40
# @Author: chenhr33733
# @File: redup.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


from simhash import Simhash, SimhashIndex
import pandas as pd
import random
from tqdm import tqdm
import pickle


def simhash_slide(inputs):
    """
    :param inputs: list of sentences
    :return:
    """
    result = []
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
    df = pd.read_json(path_or_buf='data/nbd_data/20210317_finance_text.jl', lines=True)
    inputs = df['contents'][:1000]
    result = simhash_slide(inputs)