#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/6/15 14:42
# @Author: chenhr33733
# @File: utils.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved
import compileall
import os
import re


def remove_empty_str(inputs):
    return [i for i in inputs if i]


def text_preprocess(sent):
    blank_pattern = u'( )|(\xa0)|(\u3000)|(\t)|(\n)'
    return re.sub(blank_pattern, '', sent)


def py2pyc(project_dir):
    compileall.compile_dir(project_dir)
    for root, dirs, files in os.walk(project_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
            if name[-4:] == '.pyc':
                src = os.path.join(root, name)
                dst = re.sub(r'''.cpython-38''', '', src)
                dst = re.sub(r'''/__pycache__''', '', dst)
                os.rename(src, dst)
            elif name[-3:] == '.py':
                os.remove(os.path.join(root, name))