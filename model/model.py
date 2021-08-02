# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .layers import AttentionMerge

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AlbertPreTrainedModel
from transformers import AlbertModel
import pdb
import logging

logger = logging.getLogger(__name__)

class Model(AlbertPreTrainedModel):
    """
    AlBert-AttentionMerge-Classifier

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """
    def __init__(self, config, no_att_merge=False):
        super(Model, self).__init__(config)
        self.kbert = False

        if self.kbert:
            self.albert = KBERT(config)
        else:
            self.albert = AlbertModel(config)            

        self.do_att_merge = not no_att_merge
        self.att_merge = AttentionMerge(
            config.hidden_size, 1024, 0.1) if self.do_att_merge else None
        
        self.scorer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, 1)
        )

        self.init_weights()
        self.requires_grad = {}

    def freeze_lm(self):
        logger.info('freeze lm layers.')
        for name, p in self.albert.named_parameters():
            self.requires_grad[p] = p.requires_grad
            p.requires_grad = False

    def unfreeze_lm(self):
        logger.info('unfreeze lm layers.')
        for name, p in self.albert.named_parameters():
            p.requires_grad = self.requires_grad[p]

    def score(self, h1, h2, h3, h4, h5):
        """
        h1, h2: [B, H] => logits: [B, 2]
        """
        logits1 = self.scorer(h1)
        logits2 = self.scorer(h2)
        logits3 = self.scorer(h3)
        logits4 = self.scorer(h4)
        logits5 = self.scorer(h5)
        logits = torch.cat((logits1, logits2, logits3, logits4, logits5), dim=1)
        return logits

    def forward(self, idx, input_ids, attention_mask, token_type_ids, labels):
        """
        input_ids: [B, 5, L]
        labels: [B, ]
        """
        # logits: [B, 5]

        logits = self._forward(idx, input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            logits = F.softmax(logits, dim=1)
            predicts = torch.argmax(logits, dim=1)
            right_num = torch.sum(predicts == labels)
        return loss, right_num, self._to_tensor(idx.size(0), idx.device), logits

    def _forward(self, idx, input_ids, attention_mask, token_type_ids):
        # [B, 2, L] => [B*2, L]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        
        if self.kbert:
            flat_attention_mask = self.albert.get_attention_mask()
        
        # outputs[0]: [B*5, L, H] => [B*5, H]
        if self.do_att_merge:
            h12 = self.att_merge(outputs[0], flat_attention_mask)
        else:
            h12 = outputs[0][:, 0, :]

        
        # [B*5, H] => [B*5, 1] => [B, 5]
        logits = self.scorer(h12).view(-1, 5)

        return logits

    def predict(self, idx, input_ids, attention_mask, token_type_ids):
        """
        return: [B, 5]
        """
        return self._forward(idx, input_ids, attention_mask, token_type_ids)

    def _to_tensor(self, it, device): return torch.tensor(it, device=device, dtype=torch.float)
