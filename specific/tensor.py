# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from utils.tensor import convert_to_tensor
from utils import my_dist
from .example import ConceptNetExample
import random
import logging
from torch.utils.data import DataLoader, RandomSampler, Dataset
from utils.resumable_sampler import DistributedResumableSampler
import torch.distributed as dist
from utils.feature import Feature
import torch

logger = logging.getLogger(__name__)


def make_dataloader(task, *args, **kwargs):
    assert task in ['conceptnet']
    
    if task == 'conceptnet':
        return _make_dataloader_conceptnet(*args, **kwargs)

class MCDataset(Dataset):

    def __init__(self, examples, tokenizer, max_seq_length, vary_segment_id, config, dev=False):
        # knowledge_dropout=0.0, qm_drop=0.3, am_drop=0.3, triple_drop=0.5, freq_drop=0.5
        self.examples = examples
        self.tokenizer = tokenizer
        self.dev = dev
        self.max_seq_length = max_seq_length
        self.vary_segment_id = vary_segment_id
        self.total_batch_size = config.total_batch_size
        self.seed = config.seed
        self.epoch = 0
        self.set_epoch(0) 
        # compute max length and 95% length
        max_len = 0
        all_lens = []
        def compute_len(*args):
            sum_len = 0
            for item in args:
                if item:
                    sum_len += len(item)
            return sum_len
        for example in examples:
            for token_data in example.tokens:
                this_len = compute_len(token_data['question_text'], token_data['triples_temp'],
                    token_data['qc_meaning'], token_data['ac_meaning']) + 4
                max_len = max(max_len, this_len)
                all_lens.append(this_len)
        all_lens = sorted(all_lens)
        print('max len:', max_len)
        print('95 percent len:', all_lens[int(len(all_lens)*0.95)])

    def get_state(self):
        return {'epoch': self.epoch}

    def set_state(self, state):
        self.set_epoch(state['epoch'])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.examples)
    
    def append_context(self, base, context):
        return (base + [self.tokenizer.sep_token] + context if base else context) if context else base

    def __getitem__(self, idx):
        data = self.get_example(idx)
        return convert_to_tensor(data)

    def get_example(self, idx):
        example = self.examples[idx]
        features = []
        tokens = example.tokens
        choice_mask = [0.0 for token_data in tokens]
        for c_id, token_data in enumerate(tokens):
            main_tokens = token_data['question_text']
            if token_data['is_freq_masked']:
                choice_mask[c_id] = 1.0 # 1 means masked
            if example.append_descr == 1:
                context_tokens = self.append_context([], token_data['qc_meaning'])
                context_tokens = self.append_context(context_tokens, token_data['ac_meaning'])
                context_tokens = self.append_context(context_tokens, token_data['triples_temp'])
            else:
                context_tokens = self.append_context([], token_data['triples_temp'])
            features.append(Feature.make_single(example.idx, main_tokens, context_tokens, self.tokenizer, 
            self.max_seq_length, self.vary_segment_id))
        data = ([tuple(features)], [choice_mask], [example.label], [example.is_valid])
        assert choice_mask[example.label] == 0
        return data

def _make_dataloader_conceptnet(examples, tokenizer, total_batch_size, drop_last, max_seq_length, shuffle=True, 
                                vary_segment_id=False, config=None, seed=0, dev=False):
    F = []
    L = []
    Valids = []
    for example in examples:
        example.tokenize_text(tokenizer, max_seq_length, vary_segment_id)

    dataset = MCDataset(examples, tokenizer, max_seq_length, vary_segment_id, config, dev)
    num_choices = len(examples[0].texts)
    batch_size = total_batch_size // num_choices

    sampler = DistributedResumableSampler(dataset, my_dist.get_world_size(), my_dist.get_rank(), shuffle, seed)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=batch_size, drop_last=drop_last)
    return dataloader
  
 
