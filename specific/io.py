# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from utils import _load_json, my_dist
from .example import ConceptNetExample
from datasets import load_dataset, Dataset
import logging
from copy import deepcopy
import torch.distributed as dist
from collections import Counter
import math
import random

logger = logging.getLogger(__name__)

def load_data(task, *args, **kwargs):
    assert task in {'conceptnet'}

    if task == 'conceptnet':
        return _load_data_conceptnet(*args, **kwargs)


def _load_data_conceptnet(file_name, type='json', config=None, is_train=False):
    examples = []
    append_answer_text = config.append_answer_text 
    append_descr = config.append_descr
    append_triple=(not config.no_triples)
    append_retrieval = config.append_retrieval
    sep_word = config.sep_word
    for json_obj in _load_json(file_name):
        example = ConceptNetExample.load_from_json(json_obj, append_answer_text, append_descr, 
            append_triple, append_retrieval, sep_word,
            append_frequent=config.freq_rel, frequent_thres=config.freq_threshold)
        example.is_valid = True
        examples.append(example)
    world_size = my_dist.get_world_size()
    total_len = len(examples)
    while total_len % world_size != 0:
        new_example = deepcopy(examples[0])
        new_example.is_valid = False
        examples.append(new_example)
        total_len = len(examples)        
    print(f'data: {len(examples)}, world_size: {world_size}')
    return examples

