# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from . import Vn
from . import mkdir_if_notexist

import numpy as np
import random
import torch
from . import logger
from transformers.optimization import AdamW
from torch.optim import Adamax, Adam
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.trainer_pt_utils import distributed_concat, nested_concat, nested_numpify
import json
import os
from tqdm.autonotebook import tqdm
import torch.distributed as dist
from pathlib import Path

class BaseTrainer:

    def __init__(self, model, multi_gpu, device, print_step, print_number_per_epoch, output_model_dir, vn, 
                 train_vn=None, rank=0, fp16=0, deepspeed=False, find_unused_parameters=False):
        self.fp16 = fp16        
        self.deepspeed = deepspeed
        self.model = model.to(device)
        self.lm_config = model.config
        if multi_gpu and self.fp16 in {0,1} and (not self.deepspeed):
            self.model = DDP(model, device_ids = [device], output_device=device, find_unused_parameters=find_unused_parameters)
        self.device = device
        self.multi_gpu = multi_gpu
        self.print_step = print_step
        self.print_number_per_epoch = print_number_per_epoch
        self.output_model_dir = output_model_dir
        self.current_used_last_name = 'last2'
        self.rank = rank

        self.vn = vn
        if train_vn is None:
            train_vn = vn
        self.train_record = Vn(train_vn)

    def set_optimizer(self, optimizer, scheduler=None):         
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _mean(self, tuples):
        if self.multi_gpu:
            return tuple(v.mean() for v in tuples)
        return tuples

    def _mean_sum(self, tuples):
        """
        mean if float, sum if int
        """
        if self.multi_gpu:
            return tuple(v.mean() if v.is_floating_point() else v.sum() for v in tuples)
        return tuples

    def evaluate(self, dataloader, desc='Eval', dataset_name='csqa'):
        record = Vn(self.vn)
        for batch in dataloader:
            self.model.eval()
            with torch.no_grad():
                self._forward(batch, record, dataset_name=dataset_name)
        return record

    def make_optimizer(self, weight_decay, lr, optimizer_type, warmup_proportion, t_total, num_cycles, adam_eps=1e-8):
        params = list(self.model.named_parameters())
        no_decay_keywords = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        def _no_decay(n):
            return any(nd in n for nd in no_decay_keywords)
        parameters = [
            {'params': [p for n, p in params if _no_decay(n)], 'weight_decay': 0.0},
            {'params': [p for n, p in params if not _no_decay(n)],
             'weight_decay': weight_decay}
        ]
        if optimizer_type == 'adamw':
            optimizer = AdamW(parameters, lr=lr, eps=1e-8)
        elif optimizer_type == 'adamax':
            optimizer = Adamax(parameters, lr=lr, eps=1e-8)
        elif optimizer_type == 'adam':
            optimizer = Adam(parameters, lr=lr, eps=1e-8)
        scheduler = self.make_scheduler(optimizer, warmup_proportion, t_total, num_cycles)          
        return optimizer, scheduler

    def make_scheduler(self, optimizer, warmup_proportion, t_total, num_cycles):
        if warmup_proportion == -1:
            return get_constant_schedule(optimizer)
        else:
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_proportion * t_total,
                num_training_steps=t_total, num_cycles=num_cycles)

    def save_model(self, step=None, dataloader=None, last_checkpoint=False, epoch=0):
        if last_checkpoint:
            self.current_used_last_name = 'last2' if self.current_used_last_name == 'last' else 'last'
            step = self.current_used_last_name
        output_dir = os.path.join(self.output_model_dir, str(step)) if step is not None else self.output_model_dir

        if self.rank == 0:
            logger.info('saving model to {}'.format(output_dir))
            Path(output_dir).mkdir(exist_ok=True, parents=True)
            if not self.deepspeed:
                model_to_save = self.model.module if hasattr(self.model, "module") else self.model
                model_to_save.save_pretrained(output_dir)
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                torch.save(self.scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
            random_states = {
                'random': random.getstate(),
                'np': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch.cuda': torch.cuda.get_rng_state(),
            }                        
            torch.save(random_states, os.path.join(output_dir, 'random_states.pt'))
            self.lm_config.save_pretrained(output_dir)
            if last_checkpoint:
                training_info = {
                    'global_step': self.global_step,
                    'epoch': epoch,
                    'tb_step': self.tb_step,
                }
                json.dump(training_info, open(os.path.join(output_dir, 'training_info.json'), 'w') )
        if self.multi_gpu:
            dist.barrier()
        if dataloader is not None:
            torch.save(dataloader.get_state(), os.path.join(output_dir, f'dataloader_{self.rank}.pt'))            
        if self.deepspeed:
            ds_out_dir = os.path.join(self.output_model_dir, 'deepspeed')
            self.model.save_checkpoint(ds_out_dir, str(step))
        if self.multi_gpu:
            dist.barrier()
