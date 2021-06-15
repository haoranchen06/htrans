#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 17:19
# @Author: chenhr33733
# @File: datasets.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import torch
from torch.utils.data import Dataset


class DoccanoSeq2SeqDataset(Dataset):
    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
