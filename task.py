# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from utils.base_trainer import BaseTrainer
from utils import get_device
from utils.tensor import clip_batch

import argparse
import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
import pdb
import json
from tqdm.autonotebook import tqdm

import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from apex import amp
except ImportError:
    print("apex not imported")


class Trainer(BaseTrainer):
    def __init__(self, model, multi_gpu, device, print_step,
                 output_model_dir, fp16):

        super(Trainer, self).__init__(
            model, multi_gpu, device, print_step, output_model_dir, vn=3)
        self.fp16 = fp16
        self.multi_gpu = multi_gpu
        self.tb_step = 0
        self.tb_writer = SummaryWriter(log_dir=output_model_dir)

        print("fp16 is {}".format(fp16))
            
        
    def _step(self, batch):
        loss = self._forward(batch, self.train_record)
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), 1) 
        else:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1)  # max_grad_norm = 1

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()
        self.global_step += 1
        
    def set_optimizer(self, optimizer):
        if self.fp16:
            model, optimizer = amp.initialize(self.model, optimizer, opt_level='O1')
            
            self.model = model
        self.optimizer = optimizer

    def _forward(self, batch, record):
        batch = clip_batch(batch)
        batch = tuple(t.to(self.device) for t in batch)
        all_result = self.model(*batch)
        result = all_result[:3]
        result = tuple([result[0].mean(), result[1].sum(), result[2].sum()])
        record.inc([it.item() for it in result])
        return result[0]

    def _report(self, train_record, devlp_record):
        # record: loss, right_num, all_num
        train_loss = train_record[0].avg()
        devlp_loss = devlp_record[0].avg()

        trn, tan = train_record.list()[1:]
        drn, dan = devlp_record.list()[1:]

        logger.info(f'\n____Train: loss {train_loss:.4f} {int(trn)}/{int(tan)} = {int(trn)/int(tan):.4f} |'
              f' Devlp: loss {devlp_loss:.4f} {int(drn)}/{int(dan)} = {int(drn)/int(dan):.4f}')
        self.tb_writer.add_scalar("train_accuracy", int(trn)/int(tan), global_step=self.tb_step)
        self.tb_writer.add_scalar("dev_accuracy", int(drn)/int(dan), global_step=self.tb_step)
        self.tb_writer.add_scalar("train_loss", train_loss, global_step=self.tb_step)
        self.tb_writer.add_scalar("dev_loss", devlp_loss, global_step=self.tb_step)
        self.tb_step += 1



class SelectReasonableText:
    """
    1. self.init()
    2. self.train(...)
    3. cls.load(...)
    """
    def __init__(self, config):
        self.config = config

    def init(self, ModelClass):
        gpu_ids = list(map(int, self.config.gpu_ids.split(',')))
        multi_gpu = (len(gpu_ids) > 1)
        device = get_device(gpu_ids)
        print('init_model', self.config.bert_model_dir)
        model = ModelClass.from_pretrained(self.config.bert_model_dir, cache_dir=args.cache_dir, no_att_merge=self.config.no_att_merge)
        print(model)
        if multi_gpu:
            model = nn.DataParallel(model)
        self.model = model
        self.trainer = Trainer(
            model, multi_gpu, device,
            self.config.print_step, self.config.output_model_dir, self.config.fp16)

    def train(self, train_dataloader, devlp_dataloader, save_last=True):
        t_total = len(train_dataloader) * self.config.num_train_epochs
        warmup_proportion = self.config.warmup_proportion

        optimizer = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr)
        scheduler = self.trainer.make_scheduler(optimizer, warmup_proportion, t_total)

        self.trainer.set_optimizer(optimizer)
        self.trainer.set_scheduler(scheduler)

        self.trainer.train(
            self.config.num_train_epochs, train_dataloader, devlp_dataloader, 
            save_last=save_last, freeze_lm_epochs=self.config.freeze_lm_epochs)

    def trial(self, dataloader, desc='Eval'):
        result = []
        idx = []
        labels = []
        predicts = []

        for batch in dataloader:
            batch = clip_batch(batch)
            self.model.eval()
            batch_labels = batch[4] if self.config.predict_dev else torch.zeros_like(batch[4])
            with torch.no_grad():
                all_ret = self.model(batch[0].cuda(),batch[1].cuda(),batch[2].cuda(),batch[3].cuda(),batch_labels.cuda())
                ret = all_ret[3]
                idx.extend(batch[0].cpu().numpy().tolist())
                result.extend(ret.cpu().numpy().tolist())
                labels.extend(batch[4].numpy().tolist())
                predicts.extend(torch.argmax(ret, dim=1).cpu().numpy().tolist())
        return idx, result, labels, predicts

def get_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--freeze_lm_epochs', type=int, default=0, help='freeze LM (ALBERT) until the end of this epoch (epoch number starts at 1). No freezing if =0 (default).')
    parser.add_argument('--no_att_merge',action='store_true', help='do not do attention merge, just use CLS.')

    # Path parameters
    parser.add_argument('--train_file_name', type=str, default=None)
    parser.add_argument('--devlp_file_name', type=str, default=None)
    parser.add_argument('--trial_file_name', type=str, default=None)
    parser.add_argument('--pred_file_name', type=str, default=None)
    parser.add_argument('--output_model_dir', type=str, default=None)
    parser.add_argument('--bert_model_dir', type=str, default='albert-xxlarge-v2')
    parser.add_argument('--bert_vocab_dir', type=str, default='albert-xxlarge-v2')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--predict_dir', type=str, default='output/', help='directory of prediction files.')

    # Data parameters
    parser.add_argument('--append_answer_text', type=int, default=0, help='append answer text to the question.')
    parser.add_argument('--append_descr', type=int, default=0, help='append wiktionary description.')
    parser.add_argument('--no_triples', action='store_true', help='not appending triples so we do not use ConceptNet.')


    # Other parameters
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--gpu_ids', type=str, default=None, help='default to use all gpus')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--predict_dev',action='store_true', help='predict results on dev.')
    parser.add_argument('--fp16', type=int, default=0)
    parser.add_argument('--test_mode', action='store_true')

    args = parser.parse_args()
    if args.train_file_name is None:
        args.train_file_name = os.path.join(os.environ['DATA_DIR'], 'train_data.json')
        args.devlp_file_name = os.path.join(os.environ['DATA_DIR'], 'dev_data.json')
        test_name = 'dev_data.json' if args.predict_dev else 'test_data.json'
        args.trial_file_name = os.path.join(os.environ['DATA_DIR'], test_name)
        if args.test_mode:
            print(args.train_file_name)
    pred_folder = 'dev' if args.predict_dev else 'test'
    args.predict_dir = os.path.join(args.predict_dir, pred_folder)
    Path(args.predict_dir).mkdir(exist_ok=True, parents=True)
    if args.pred_file_name is not None:
        args.pred_file_name = os.path.join(args.predict_dir, args.pred_file_name)
        print('output to:', args.pred_file_name)
    if args.output_model_dir is None:
        args.output_model_dir = os.path.join(os.environ['OUTPUT_DIR'], 'model')
        if args.test_mode:
            print(args.output_model_dir)
    if args.cache_dir is None:
        args.cache_dir = os.path.join(os.environ['DATA_DIR'], 'model')
        if args.test_mode:
            print(args.cache_dir)

    return args


if __name__ == '__main__':
    import time
    start = time.time()
    print("start is {}".format(start))
    import random
    import numpy as np

    from transformers import AlbertTokenizer
    from transformers import AlbertConfig

    from specific.io import load_data
    from pathlib import Path
    from specific.tensor import make_dataloader
    from model.model import Model
    from utils.common import mkdir_if_notexist

    args = get_args()
    if args.gpu_ids is None:
        n_gpus = torch.cuda.device_count()
        args.gpu_ids=','.join([str(i) for i in range(n_gpus)])
        print('gpu_ids:', args.gpu_ids)


    args.fp16 = True if args.fp16 == 1 else False
    
    
    print("args.fp16 is {}".format(args.fp16))
    assert args.mission in ('train', 'output')

    # ------------------------------------------------#
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # ------------------------------------------------#

    # ------------------------------------------------#
    experiment = 'conceptnet'
    if args.mission == 'train':
        print('load_data', args.train_file_name)
        train_data = load_data(experiment, args.train_file_name, type='json', append_answer_text=args.append_answer_text, 
            append_descr=args.append_descr, append_triple=(not args.no_triples))

        print('load_data', args.devlp_file_name)
        devlp_data = load_data(experiment, args.devlp_file_name, type='json', append_answer_text=args.append_answer_text, 
            append_descr=args.append_descr, append_triple=(not args.no_triples))
        if args.test_mode:
            print('test mode')
            train_data = train_data[:80]
            devlp_data = devlp_data[:80]
            args.print_step = 10
    elif args.mission == 'output':
        print('load_data', args.trial_file_name)
        devlp_data = load_data(experiment, args.trial_file_name, type='json', append_answer_text=args.append_answer_text, 
            append_descr=args.append_descr, append_triple=(not args.no_triples))
    print('get dir {}'.format(args.output_model_dir))
    Path(args.output_model_dir).mkdir(exist_ok=True, parents=True)
    print('load_vocab', args.bert_vocab_dir)
    tokenizer = AlbertTokenizer.from_pretrained(args.bert_vocab_dir)

    log_file = time.strftime("%Y-%m-%d-%H-%M-%S.log", time.gmtime())
    fh = logging.FileHandler(os.path.join(args.output_model_dir, log_file))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # ------------------------------------------------#

    # ------------------------------------------------#
    print('make dataloader ...')
    if args.mission == 'train':
        train_dataloader = make_dataloader(
            experiment, train_data, tokenizer, batch_size=args.batch_size,
            drop_last=False, max_seq_length=args.max_seq_length)  # 52 + 3

        print('train_data %d ' % len(train_data))

    devlp_dataloader = make_dataloader(
            experiment, devlp_data, tokenizer, batch_size=args.batch_size,
            drop_last=False, max_seq_length=args.max_seq_length, shuffle=False)

    print('devlp_data %d ' % len(devlp_data))
    # ------------------------------------------------#

    # -------------------- main ----------------------#
    if args.mission == 'train':
        srt = SelectReasonableText(args)
        srt.init(Model)
        srt.train(train_dataloader, devlp_dataloader, save_last=False)

        srt = SelectReasonableText
    elif args.mission == 'output':
        srt = SelectReasonableText(args)
        srt = SelectReasonableText(args)
        srt.init(Model)
        idx, result, label, predict = srt.trial(devlp_dataloader)

        content = ''
        length = len(result)
        right = 0
        for i, item in enumerate(tqdm(result)):
            if predict[i] == label[i]:
                right += 1
            content += '{},{},{},{},{},{},{},{}\n' .format(idx[i][0], item[0], item[1], item[2], item[3], item[4], label[i], predict[i])

        res_data = {'idx': idx, 'result': result, 'label': label, 'predict': predict}
        logger.info("accuracy is {}".format(right/length))
        with open(args.pred_file_name, 'w', encoding='utf-8') as f:
            f.write(content)    
        with open(args.pred_file_name.replace('.csv', '.json'), 'w', encoding='utf-8') as f:
            json.dump(res_data, f)
        with open(args.pred_file_name.replace('.csv', '_summary.json'), 'w', encoding='utf-8') as f:
            summary_data = {'correct': right, 'total': length}
            json.dump(summary_data, f)



            # ------------------------------------------------#
    
    end = time.time()
    logger.info("start is {}, end is {}".format(start, end))
    logger.info("Running time:%.2fseconds."%(end-start))
