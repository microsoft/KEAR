# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch.distributed as dist

def get_world_size():
    try:
        return dist.get_world_size()
    except:
        return 1

def get_rank():
    try:
        return dist.get_rank()
    except:
        return 0