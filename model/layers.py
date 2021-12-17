# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
        
class ChoicePredictor(nn.Module):
    def __init__(self, config, opt):
        super(ChoicePredictor, self).__init__()
        final_dropout_prob = opt['final_pred_dropout_prob'] # add a parameter here to ensure backward safety
        self.dropout = nn.Dropout(final_dropout_prob)
        out_dim = config.hidden_size
        self.scorer = nn.Linear(out_dim, 1)
        self.config = config
        self.my_config = opt
    
    def forward(self, outputs, attention_mask):
        h12 = outputs[0][:, 0, :]
        h12 = self.dropout(h12)
        num_choices = attention_mask.size(1)
        h12 = h12.view(-1, num_choices, self.config.hidden_size)
        logits = self.scorer(h12).view(-1, num_choices)
        return logits
