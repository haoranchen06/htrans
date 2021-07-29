#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time: 2021/7/28 15:16
# @Author: chenhr33733
# @File: ghm_loss.py
# @Software: PyCharm
# @Copyright：Copyright(c) 2021 Hundsun.com,Inc.All Rights Reserved


import torch
from torch import nn
import torch.nn.functional as F


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


class GHMLoss(nn.Module):
    def __init__(self, bins, alpha):
        super(GHMLoss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self._last_bin_count = None

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        g = torch.abs(self._custom_loss_grad(x, target)).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros((self._bins))
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = (x.size(0) * x.size(1))

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count

        nonempty_bins = (bin_count > 0).sum().item()

        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=0.0001)
        beta = N / gd

        return self._custom_loss(x, target, beta[bin_idx])


class GHMCLoss(GHMLoss):
    def __init__(self, bins, alpha):
        super(GHMCLoss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        return F.binary_cross_entropy_with_logits(x, target, weight=weight)

    def _custom_loss_grad(self, x, target):
        return torch.sigmoid(x).detach() - target


class GHMRLoss(GHMLoss):
    def __init__(self, bins, alpha, mu):
        super(GHMRLoss, self).__init__(bins, alpha)
        self._mu = mu

    def _custom_loss(self, x, target, weight):
        d = x - target
        mu = self._mu
        loss = torch.sqrt(d * d + mu * mu) - mu
        N = x.size(0) * x.size(1)
        return (loss * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        d = x - target
        mu = self._mu
        return d / torch.sqrt(d * d + mu * mu)


class GHMCELoss(GHMLoss):
    def __init__(self, bins, alpha):
        super(GHMCELoss, self).__init__(bins, alpha)

    def _custom_loss(self, x, target, weight):
        target = target.view(-1, 1)
        log_prob = F.log_softmax(x, dim=1)
        log_prob = log_prob.gather(1, target)
        weight = weight.to(log_prob.device)
        weighted_loss = torch.mean(-log_prob * weight)
        return weighted_loss

    def _custom_loss_grad(self, x, target):
        prob = F.softmax(x, dim=1)
        expect = one_hot(target, x.size(1)).to(prob.device)
        grad = (prob - expect).mean(dim=1)
        return grad


if __name__ == '__main__':
    loss_fct = GHMCELoss(bins=10, alpha=1)
    a = torch.tensor([[0., 50000, -5], [1., 0.5, 0.]])
    b = torch.tensor([1, 2])
    loss = loss_fct(x=a, target=b)
