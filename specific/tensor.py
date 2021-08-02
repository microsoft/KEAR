# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.tensor import convert_to_tensor


def make_dataloader(task, *args, **kwargs):
    if task == 'conceptnet':
        return _make_dataloader_conceptnet(*args, **kwargs)

def _make_dataloader_conceptnet(examples, tokenizer, batch_size, drop_last, max_seq_length, shuffle=True):
    F = []
    L = []

    for example in examples:
        f1, f2, f3, f4, f5, la = example.fl(tokenizer, max_seq_length)

        F.append((f1, f2, f3, f4, f5))
        L.append(la)

    return convert_to_tensor((F, L), batch_size, drop_last, shuffle=shuffle)
 