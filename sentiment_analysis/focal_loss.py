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
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        log_pt = F.log_softmax(inputs)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = Variable(log_pt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == '__main__':
    loss_fct = FocalLoss()
    a = torch.Tensor(3)
    b = torch.tensor(1)
    res = loss_fct(a, b)