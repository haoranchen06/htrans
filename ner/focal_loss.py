#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/7/21 11:06
# @Author: chenhr33733
# @File: focal_loss.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, inputs, target):
        """
        :param inputs: torch.tensor with dim of 2, batch_size * hidden_dim
        :param target: torch.tensor
        :return:
        """
        if inputs.dim() == 1:
            inputs = inputs.view(1, -1)
        target = target.view(-1, 1)

        log_pt = F.log_softmax(inputs, dim=1)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = Variable(log_pt.data.exp())

        if self.alpha is not None:
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    fl_fct = FocalLoss(gamma=2)
    cl_fct = torch.nn.CrossEntropyLoss()
    a = torch.tensor([[0., 1., 0.], [1., 0., 0.]])
    b = torch.tensor([1, 0])
    fl = fl_fct(a, b)
    cl = cl_fct(a, b)
