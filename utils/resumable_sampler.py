# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from torch.utils.data.sampler import Sampler
import torch
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
import datasets

        

class DistributedResumableSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        # number of data must be divisible of num_replicas
        super(DistributedResumableSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed)
        self.current_idx = rank
        self.perm = None
        self.current_len = self.num_samples

        
    def set_epoch(self, epoch):
        super(DistributedResumableSampler, self).set_epoch(epoch)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            self.perm = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            self.perm = list(range(len(self.dataset)))
        assert len(self.perm) == self.total_size
        self.current_idx = self.rank

        while self.current_idx < self.total_size:
            self.current_idx += self.num_replicas
            yield self.perm[self.current_idx - self.num_replicas]
        
        self.current_len = self.num_samples

    def __len__(self):
        return self.current_len

    def get_state(self):
        return {"perm": self.perm, "current_idx": self.current_idx, 'rank': self.rank, 'dataset': self.dataset.get_state()}
    
    def load_state(self, state):
        self.perm = state["perm"]
        self.current_idx = state["current_idx"]
        print('current_idx:', self.current_idx)
        self.current_len = self.num_samples - (self.total_size - self.current_idx) // self.num_replicas
        assert self.rank == state['rank']
        self.dataset.set_state(state['dataset'])

if __name__ == '__main__':
    dataset = [1,2,3,4,5,6,7]
    sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
    for b in sampler:
        print(b)

