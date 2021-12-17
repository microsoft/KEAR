# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from utils.base_trainer import BaseTrainer
from utils import get_device
from utils.gatherer import Gatherer
from utils.tensor import clip_batch
from torch.cuda.amp import autocast, GradScaler
import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm.autonotebook import tqdm
import random
from collections import defaultdict
import shutil
import numpy as np
import torch.distributed as dist
import torch.autograd.profiler as profiler
from . import Vn
from utils import my_dist
import deepspeed
import time
import logging
logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, print_step, print_number_per_epoch, 
                 output_model_dir, fp16, clip_batch=True, start_training_epoch=0, training_records=None,
                 save_interval_step=100, start_tb_step=0, print_loss_step=None, config=None):
        vn = 3
        train_vn = vn+1 if config.adv_train else vn
        self.config = config
        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, print_number_per_epoch, output_model_dir, vn=vn, train_vn=train_vn, 
            rank=config.local_rank, fp16=fp16, deepspeed=config.deepspeed, find_unused_parameters=config.model_type=='roberta')
        if self.fp16 == 1:
            self.scaler = GradScaler()
        self.multi_gpu = multi_gpu
        self.tb_step = start_tb_step
        self.tb_writer = SummaryWriter(log_dir=output_model_dir)
        self.gatherer = Gatherer(dist.get_world_size()) if multi_gpu else None
        self.clip_batch = clip_batch
        self.best_dev_perf = defaultdict(float)
        self.test_debug = False
        if training_records:
            self.training_records = training_records
        else:
            self.training_records = {
                'performance': {}, # global_step -> perf_records
                'epoch': {}, # epoch_num -> global_step
            }
        if 'current_used_last_name' in self.training_records:
            self.current_used_last_name = self.training_records['current_used_last_name']
        self.start_training_epoch = start_training_epoch
        self.save_interval_step = save_interval_step
        self.print_loss_step = print_loss_step if print_loss_step is not None else self.print_step
        print("Trainer: fp16 is {}".format(self.fp16))
            
    def train(self, epoch_num, train_dataloader, dev_dataloaders, 
              save_last=True, save_every=False, start_global_step=0):

        best_dev_loss = float('inf')
        self.global_step = start_global_step
        self.save_every = save_every
        if len(self.training_records['performance']) > 0:
            last_saved_step = max([int(k) for k in self.training_records['performance'].keys()])
            for key, val in self.training_records['performance'][str(last_saved_step)].items():
                if key.startswith('best_dev_accuracy_'):
                    dataset_name = key.split('best_dev_accuracy_')[1]
                    self.best_dev_perf[dataset_name] = val
            print('loaded best accuracy.', self.best_dev_perf)
        self.train_record.init()
        self.model.zero_grad()
        evaluate_step = self.print_step if self.print_number_per_epoch is None else len(train_dataloader) // self.print_number_per_epoch
        logger.info(f"total n_step = {len(train_dataloader)}, evaluate_step = {evaluate_step}")
        current_train_dataloader = train_dataloader
        for epoch in range(self.start_training_epoch, int(epoch_num)):
            print(f'---- Epoch: {epoch+1:02} ----')
            train_looper = tqdm(current_train_dataloader, desc='Train') if self.config.test_mode else current_train_dataloader
            for step, batch in enumerate(train_looper):
                self.model.train()
                loss = self._step(batch)
                if self.global_step % self.print_loss_step == 0:
                    print(f'global_step={self.global_step}, loss={loss}')
                    print(f'gpu {torch.cuda.current_device()}: memory {torch.cuda.max_memory_allocated()/1024/1024}')
                if self.global_step % evaluate_step == 0:
                    logger.info(f'step = {step}, global_step = {self.global_step}, evaluate_step = {evaluate_step}')
                    dev_records = {}
                    for dataset_name, dev_dataloader in dev_dataloaders.items():
                        print(f'evaluating dataset {dataset_name}')
                        dev_records[dataset_name] = self.evaluate(dev_dataloader, dataset_name=dataset_name)
                    if save_every:
                        print('save every epoch.')
                        self.save_model(self.global_step)
                    self._report(self.train_record, dev_records)
                    self.train_record.init()
                if self.save_interval_step is not None and self.global_step % self.save_interval_step == 0:
                    self.save_model(dataloader=current_train_dataloader, last_checkpoint=True, epoch=epoch)
                    self.save_training_record(last_checkpoint=True)
        dev_records = {}
        for dataset_name, dev_dataloader in dev_dataloaders.items():
            dev_records[dataset_name] = self.evaluate(dev_dataloader, dataset_name=dataset_name)
        self._report(self.train_record, dev_records)

        if save_last:
            self.save_model(self.global_step)

    def _step(self, batch):
        loss, _ = self._forward(batch, self.train_record, mode='train', dataset_name=self.config.data_version)
        if self.deepspeed:
            self.model.backward(loss)
        elif self.fp16 == 1:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()         
        self.global_step += 1
        if self.global_step % self.config.gradient_acc_step == 0:
            if self.deepspeed:
                self.model.step()
            else:
                if self.fp16 == 1:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1)  # max_grad_norm = 1               
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1)
                    self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
        return loss

    def deepspeed_wrap(self, optimizer=None, scheduler=None):
        if self.deepspeed:
            print('initializing')
            model, optimizer, _, scheduler = deepspeed.initialize(args=self.config, model=self.model, optimizer=optimizer, 
                                            lr_scheduler=scheduler, config=self.config.deepspeed_config)
            print('initialize finish')
            self.model = model
        return optimizer, scheduler    

    def deepspeed_init_inference(self, optimizer=None, scheduler=None):
        if self.deepspeed:
            self.deepspeed_wrap(optimizer, scheduler)
            ds_path = os.path.join(self.config.bert_model_dir, 'deepspeed')
            print('ds_path:', ds_path)
            load_path, _ = self.model.load_checkpoint(ds_path, 'None')
            assert load_path is not None            
        
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _forward(self, batch, record, dataset_name=None, mode='dev', return_all=False):
        if dataset_name is None:
            dataset_name = batch[-1]
        else:
            batch = (*batch, dataset_name)
        valids = batch[-2]
        batch = tuple(list(batch[:-2]) + [batch[-1]])
        valids = valids.to(self.device)
        if self.clip_batch:
            batch = clip_batch(batch)
        moved_batch = []
        for t in batch:
            moved_batch.append(str(t) if (type(t) is np.str_ or type(t) is str) else t.to(self.device))
        batch = tuple(moved_batch)
        if self.fp16 == 1:
            with autocast():
                all_result = self.model(*batch, mode)
        else:
            all_result = self.model(*batch, mode)
        if return_all:
            return all_result
        loss, right_num, input_size, logits, adv_norm = all_result 
        if self.multi_gpu:
            valids_all = self.gatherer.gather_all(valids)
            loss_all = self.gatherer.gather_valid(loss, valids_all)
            loss_all = loss_all.mean()
            right_num_all = self.gatherer.gather_valid(right_num, valids_all).sum()
            valid_num = valids_all.sum()
            result = [loss_all, right_num_all, valid_num]
            if adv_norm is not None:
                adv_norm = adv_norm.view(-1)
                adv_norm_all = self.gatherer.gather_all(adv_norm).mean()
                result.append(adv_norm_all)
            record.inc([it.item() for it in result])
            loss = loss.mean() / self.config.gradient_acc_step
        else:
            loss = loss.mean() / self.config.gradient_acc_step
            if adv_norm is not None:
                result = tuple([loss, right_num.sum(), input_size.sum(), adv_norm.mean()])
            else:
                result = tuple([loss, right_num.sum(), input_size.sum()])
            record.inc([it.item() for it in result])                
        return loss, all_result
    
    def save_training_record(self, last_checkpoint=False):
        self.training_records['current_used_last_name'] = self.current_used_last_name
        if self.rank == 0:
            if last_checkpoint:
                json.dump(self.training_records, open(os.path.join(self.output_model_dir, self.current_used_last_name, 'training_records.json'), 'w')) # save a copy to last checkpoint
            json.dump(self.training_records, open(os.path.join(self.output_model_dir, 'training_records.json'), 'w'))

    def _report(self, train_record, devlp_records, epoch=None):
        # record: loss, right_num, all_num
        record_dict = {}
        train_loss = train_record[0].avg()
        trn, tan = train_record.list()[1:3]
        if tan == 0:
            tan = 1
        logger.info(f'____Train: loss {train_loss:.4f} {int(trn)}/{int(tan)} = {int(trn)/int(tan):.4f} |')
        if len(train_record) >= 4:
            adv_norm = train_record[3].avg()
            logger.info(f'____ adv_norm {adv_norm:.4f}')
            record_dict['adv_norm'] = adv_norm
        record_dict["train_accuracy"] = int(trn)/int(tan)
        record_dict["train_loss"] = train_loss
        target_dataset_name = list(devlp_records.keys())[0]
        for dataset_name, devlp_record in devlp_records.items():
            devlp_loss = devlp_record[0].avg()
            drn, dan = devlp_record.list()[1:]
            if dan == 0:
                dan = 1

            logger.info(f' Devlp {dataset_name}: loss {devlp_loss:.4f} {int(drn)}/{int(dan)} = {int(drn)/int(dan):.4f}')
            if not self.save_every and dataset_name == target_dataset_name:
                current_acc = int(drn) / int(dan)
                logger.info(f'save best: current acc = {current_acc}, best: {self.best_dev_perf[dataset_name]}')
                if current_acc > self.best_dev_perf[dataset_name]:
                    print('saving best model.')
                    self.save_model()
                    self.best_dev_perf[dataset_name] = current_acc
            else:
                self.best_dev_perf[dataset_name] = max(self.best_dev_perf[dataset_name], int(drn) / int(dan))

            record_dict[f"dev_accuracy_{dataset_name}"] = int(drn)/int(dan)
            record_dict[f"best_dev_accuracy_{dataset_name}"] = self.best_dev_perf[dataset_name]
            record_dict[f"dev_loss_{dataset_name}"] = devlp_loss
        if self.rank == 0:
            for name, val in record_dict.items():
                self.tb_writer.add_scalar(name, val, global_step=self.tb_step)
        self.training_records['performance'][self.global_step] = record_dict
        if epoch is not None:
            self.training_records['epoch'][str(epoch)] = self.global_step
        if self.rank == 0:
            self.save_training_record()
        self.tb_step += 1