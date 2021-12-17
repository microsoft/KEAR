# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from transformers.tokenization_utils_base import BatchEncoding

class DataLoaderSampler:
    '''
    Takes in a dataloader and sampler, makes a resumable dataloader.
    '''
    def __init__(self, dataloader, dataset_name, seed=None):
        self.dataset_name = dataset_name
        self.effective_length = len(dataloader)
        self.dataloader = dataloader
        self.current_length = self.effective_length
        self.idx = 0
        if seed is None:
            self.rng = np.random
        else:
            self.rng = np.random.default_rng(seed=seed)
        print(f'total length: {self.effective_length}')
        self.dataloader_iter = iter(dataloader)

    def __len__(self):
        return self.current_length
    
    def __iter__(self):      
        while self.idx < self.effective_length:
            this_dataset = self.dataset_name
            self.idx += 1
            try:
                batch = next(self.dataloader_iter)
            except StopIteration: # dataloader ends, start another one
                self.dataloader_iter = iter(self.dataloader)
                batch = next(self.dataloader_iter)
            yield batch
        self.current_length = self.effective_length
        self.idx = 0
        self.dataloader_iter = iter(self.dataloader)

    def set_epoch(self, epoch):
        for dataloader in self.dataloaders.values():
            dataloader.sampler.set_epoch(epoch)

    def load_state(self, state):
        self.idx = state['idx']
        self.rng.set_state(state['rng_state'])
        self.dataloader.sampler.load_state(state['dataloader_sampler'])
        self.current_length = self.effective_length - self.idx
    
    def get_state(self):
        return {
            'idx': self.idx,
            'rng_state': self.rng.get_state(),
            'dataloader_sampler': self.dataloader.sampler.get_state(),
        }


        
        
    



        