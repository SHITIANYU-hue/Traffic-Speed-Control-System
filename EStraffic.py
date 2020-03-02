#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:49:49 2018

@author: traffic203
"""

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space, hidden_neuron):
        """
        Really I should be using inheritance for the small_net here
        """
        super(ES, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = action_space
        self.hidden_neuron = hidden_neuron
        self.linear = []
        self.linear1 = nn.Linear(self.num_inputs, self.hidden_neuron)
        self.linear2 = nn.Linear(self.hidden_neuron, self.hidden_neuron)
        self.actor_linear = nn.Linear(self.hidden_neuron, self.num_outputs)
        self.train()

    def forward(self, inputs):
        x = selu(self.linear1(inputs))
        x = selu(self.linear2(x))
        return torch.sigmoid(self.actor_linear(x))


    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]