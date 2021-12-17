# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



from .feature import Feature
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from .resumable_sampler import DistributedResumableSampler


def clip_batch(batch):
    """
    clip batch based on max length
    """
    input_ids = batch[1]
    attention_mask = batch[2]
    token_type_ids = batch[3]
    question_mask = batch[4]
    batch_size = input_ids.size(0)
    num_dim = len(input_ids.size())
    while True:
        end_flag = False
        for i in range(batch_size):
            this_loc = input_ids[i, :, -1].any() if num_dim == 3 else input_ids[i, -1]
            if this_loc != 0:
                end_flag = True
                break
        if end_flag:
            break
        else:
            input_ids = input_ids[:, :, :-1] if num_dim == 3 else input_ids[:, :-1]
    max_seq_length = input_ids.size(-1)
    if num_dim == 3:
        attention_mask = attention_mask[:, :, :max_seq_length]
        token_type_ids = token_type_ids[:, :, :max_seq_length]
        question_mask = question_mask[:, :, :max_seq_length]
    elif num_dim == 2:
        attention_mask = attention_mask[:, :max_seq_length]
        token_type_ids = token_type_ids[:, :max_seq_length]
        question_mask = question_mask[:, :max_seq_length]

    batch = tuple([batch[0]] + [input_ids, attention_mask, token_type_ids, question_mask] + list(batch[5:]))
    return batch

def convert_to_tensor(data, squeeze_first=True):
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
        
        elif type(item[0]) is bool:
            _tensor = torch.tensor(item, dtype=torch.bool)
            tensors.append(_tensor)


        elif type(item[0]) is list:
            if type(item[0][0]) is int:
                _tensor = torch.tensor(item, dtype=torch.long)
            elif type(item[0][0]) is float:
                _tensor = torch.tensor(item, dtype=torch.float)
            try:
                tensors.append(_tensor)
            except:
                breakpoint()

        else:
            raise Exception(str(type(item[0])))

    if squeeze_first:
        tensors = [tensor.squeeze(0) for tensor in tensors]
    return tuple(tensors)


def _convert_feature_to_tensor(features):
    """
    features: [f, f, f, ...]
    """
    all_idx = torch.tensor([f.idx for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.float)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_question_mask = torch.tensor([f.question_mask for f in features], dtype=torch.float)
    if features[0].pot_spans is not None:
        all_pot_spans = torch.tensor([f.pot_spans for f in features], dtype=torch.long)
        all_label_span = torch.tensor([f.label_span for f in features], dtype=torch.long)
        return all_idx, all_input_ids, all_input_mask, all_segment_ids, all_question_mask, all_pot_spans, all_label_span
    else:
        return all_idx, all_input_ids, all_input_mask, all_segment_ids, all_question_mask


def _convert_multi_feature_to_tensor(features):
    """
    features: [(f1, f2, f3), ...]
    """
    all_idx = torch.tensor([[f.idx for f in fs] for fs in features], dtype=torch.long)
    all_input_ids = torch.tensor([[f.input_ids for f in fs] for fs in features], dtype=torch.long)
    all_input_mask = torch.tensor([[f.input_mask for f in fs] for fs in features], dtype=torch.float)
    all_segment_ids = torch.tensor([[f.segment_ids for f in fs] for fs in features], dtype=torch.long)
    all_question_mask = torch.tensor([[f.question_mask for f in fs] for fs in features], dtype=torch.float)
    return all_idx, all_input_ids, all_input_mask, all_segment_ids, all_question_mask
