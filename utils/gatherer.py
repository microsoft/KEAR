# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.distributed as dist

class Gatherer():

    def __init__(self, world_size):
        self.world_size = world_size
    
    def gather_all(self, tensor):
        tensor_list = [torch.zeros_like(tensor) for idx in range(self.world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list)
    
    def gather_valid(self, tensor, valid_all):
        # tensor: not gathered
        # valid_all: gathered
        tensor_all = self.gather_all(tensor)
        return torch.masked_select(tensor_all, valid_all)



