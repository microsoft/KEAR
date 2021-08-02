# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.



from .feature import Feature

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


def clip_batch(batch):
    """
    clip batch based on max length
    """
    # print("batch size is {}".format(len(batch[0])))
    idx, input_ids, attention_mask, token_type_ids, labels = batch
    # [batch_size, 2, L]
    batch_size = input_ids.size(0)
    while True:
        end_flag = False
        for i in range(batch_size):
            if input_ids[i, 0, -1] != 0:
                end_flag = True
            if input_ids[i, 1, -1] != 0:
                end_flag = True 
        
        if end_flag:
            break
        else:
            input_ids = input_ids[:, :, :-1]
    
    max_seq_length = input_ids.size(2)
    attention_mask = attention_mask[:, :, :max_seq_length]
    token_type_ids = token_type_ids[:, :, :max_seq_length]
    
    return idx, input_ids, attention_mask, token_type_ids, labels

def convert_to_tensor(data, batch_size, drop_last, shuffle):
    tensors = []

    for item in data:
        # item: (f, f, f, ...)
        # item: ((f1, f2, f3), ...)
        # item: (int, int, int, ...)
        # item: ((int, int, int), ...)
        # item: ((float, float, float), ...)
        if type(item[0]) is Feature:
            _tensors = _convert_feature_to_tensor(item)
            tensors.extend(_tensors)

        elif type(item[0]) is tuple:
            if type(item[0][0]) is Feature:
                _tensors = _convert_multi_feature_to_tensor(item)
                tensors.extend(_tensors)

        elif type(item[0]) is int:
            _tensor = torch.tensor(item, dtype=torch.long)
            tensors.append(_tensor)

        elif type(item[0]) is list:
            if type(item[0][0]) is int:
                _tensor = torch.tensor(item, dtype=torch.long)
            elif type(item[0][0]) is float:
                _tensor = torch.tensor(item, dtype=torch.float)

            tensors.append(_tensor)

        else:
            raise Exception(str(type(item[0])))

    dataset = TensorDataset(*tensors)

    sampler = RandomSampler(dataset) if shuffle else None
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=batch_size, drop_last=drop_last)
    return dataloader


def _convert_feature_to_tensor(features):
    """
    features: [f, f, f, ...]
    """
    all_idx = torch.tensor([f.idx for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    return all_idx, all_input_ids, all_input_mask, all_segment_ids


def _convert_multi_feature_to_tensor(features):
    """
    features: [(f1, f2, f3), ...]
    """
    all_idx = torch.tensor([[f.idx for f in fs] for fs in features], dtype=torch.long)
    all_input_ids = torch.tensor([[f.input_ids for f in fs] for fs in features], dtype=torch.long)
    all_input_mask = torch.tensor([[f.input_mask for f in fs] for fs in features], dtype=torch.long)
    all_segment_ids = torch.tensor([[f.segment_ids for f in fs] for fs in features], dtype=torch.long)
    return all_idx, all_input_ids, all_input_mask, all_segment_ids
