from utils.base_trainer import BaseTrainer
from utils import get_device
from utils.tensor import clip_batch
from torch.cuda.amp import autocast, GradScaler
import sys
import argparse
import torch
from torch import nn
import os
from torch.utils.tensorboard import SummaryWriter
import json
from tqdm.autonotebook import tqdm
import random
from utils.dataloader_sampler import DataLoaderSampler
from collections import defaultdict
import shutil
from utils.trainer import Trainer
from utils import my_dist
import deepspeed
from transformers import AutoConfig

import logging; logging.getLogger("transformers").setLevel(logging.WARNING)
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelectReasonableText:
    """
    1. self.init()
    2. self.train(...)
    3. cls.load(...)
    """
    def __init__(self, config):
        self.config = config

    def init(self, ModelClass):
        multi_gpu = self.config.ddp
        device = torch.device(self.config.local_rank)
        print('init_model', self.config.bert_model_dir)
        ModelClass.set_config(self.config.model_type)
        print('deepspeed:', self.config.deepspeed)
        print('resume_training:', self.config.resume_training)
        if self.config.deepspeed and (self.config.resume_training or self.config.mission == 'output'):
            config_path = self.config.bert_model_dir
            print(f'config_path:{config_path}')
            lm_config = AutoConfig.from_pretrained(config_path) 
            model = ModelClass(lm_config, opt=vars(self.config))
        else:
            load_dir = self.config.bert_model_dir
            model = ModelClass.from_pretrained(load_dir, cache_dir=args.cache_dir, opt=vars(self.config))

        if multi_gpu:
            dist.barrier()
        self.model = model
        logger.info('initializing trainer.')
        self.trainer = Trainer(
            model, multi_gpu, device,
            self.config.print_step, self.config.print_number_per_epoch, self.config.output_model_dir, self.config.fp16,
            clip_batch=not self.config.test_mode, start_training_epoch=self.config.start_training_epoch,
            training_records=self.config.training_records, save_interval_step=self.config.save_interval_step,
            start_tb_step=self.config.start_tb_step, print_loss_step=self.config.print_loss_step, config=self.config)
        logger.info('initialize trainer finished.')

    def train(self, train_dataloader, devlp_dataloaders, save_last=True, save_every=False):
        t_total = len(train_dataloader) * self.config.num_train_epochs // self.config.gradient_acc_step
        warmup_proportion = self.config.warmup_proportion

        logger.info('setting up optimizer')
        optimizer, scheduler = self.trainer.make_optimizer(self.config.weight_decay, self.config.lr, self.config.optimizer_type, 
                                                warmup_proportion, t_total, self.config.scheduler_num_cycles, self.config.adam_eps)
        logger.info('deepspeed wrap')
        optimizer, scheduler = self.trainer.deepspeed_wrap(optimizer, scheduler)
        print('finish deepspeed wrap')
        if self.config.load_training_dir:
            if self.config.deepspeed:
                ds_path = os.path.join(self.trainer.output_model_dir, 'deepspeed')
                load_path, _ = self.trainer.model.load_checkpoint(ds_path, self.config.loading_used_name)
                assert load_path is not None
            else:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config.local_rank}
                optimizer.load_state_dict(torch.load(os.path.join(args.load_training_dir, "optimizer.pt"), map_location=map_location))
                scheduler.load_state_dict(torch.load(os.path.join(args.load_training_dir, "scheduler.pt"), map_location=map_location))
            random_states = torch.load(os.path.join(args.load_training_dir, "random_states.pt"))
            random.setstate(random_states['random'])
            np.random.set_state(random_states['np'])
            torch.set_rng_state(random_states['torch'])
            torch.cuda.set_rng_state(random_states['torch.cuda'])
            train_dataloader.load_state(torch.load(os.path.join(args.load_training_dir, f'dataloader_{self.config.local_rank}.pt')))

        if self.config.ddp:
            dist.barrier()
        print('load successfully.')                                      
        self.trainer.set_optimizer(optimizer, scheduler)
        self.trainer.train(
            self.config.num_train_epochs, train_dataloader, devlp_dataloaders, 
            save_last=save_last, save_every=save_every, start_global_step=self.config.start_global_step)  

    def trial(self, dataloader, desc='Eval'):
        using_dataset_name = self.config.data_version
        logger.info('setting up optimizer')
        self.trainer.deepspeed_init_inference()
        result = []
        idx = []
        labels = []
        predicts = []
        looper = tqdm(dataloader, desc='Predict') if self.config.local_rank == 0 else dataloader
        for batch in looper:
            batch = clip_batch(batch)
            self.model.eval()
            this_label = batch[-2]
            if batch[-2].sum() < 0:
                batch = list(batch[:-2]) + [torch.zeros_like(batch[-2]), batch[-1]]
                batch = tuple(batch)
            with torch.no_grad():
                loss, right_num, input_size, logits, adv_loss = self.trainer._forward(batch, None, mode='dev', dataset_name=using_dataset_name, return_all=True)
                idx.extend(batch[0].cpu().numpy().tolist())
                result.extend(logits.cpu().numpy().tolist())
                labels.extend(this_label.numpy().tolist())
                predicts.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        return idx, result, labels, predicts

def get_args():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_batch_size', type=int, default=None, help='number of choices in a batch.')
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--scheduler_num_cycles', type=int, default=1)
    parser.add_argument('--optimizer_type', type=str, default='adamw', help='adamw or adamax.')
    parser.add_argument('--gradient_acc_step', type=int, default=1, help='gradient accumulation step.')
    parser.add_argument('--gradient_acc_batch_size',type=int, default=None, help='target batch size for gradient acc. Overrides gradient_acc_step.')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_seq_length', type=int, default=64)
    parser.add_argument('--final_pred_dropout_prob', type=float, default=0., help='dropout prob of prediction layer.')
   
    # for DDP
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ddp", action='store_true', help='use ddp.')

    # Path parameters
    parser.add_argument('--train_file_name', type=str, default=None)
    parser.add_argument('--devlp_file_name', type=str, default=None)
    parser.add_argument('--trial_file_name', type=str, default=None)
    parser.add_argument('--data_version', type=str, default=None, help='data version for CSQA.')
    parser.add_argument('--pred_file_name', type=str, default=None)
    parser.add_argument('--output_model_dir', type=str, default=None)
    parser.add_argument('--bert_model_dir', type=str, default='albert-xxlarge-v2')
    parser.add_argument('--bert_vocab_dir', type=str, default='albert-xxlarge-v2')
    parser.add_argument('--model_type', type=str, default='albert', help='albert, deberta, electra, roberta.')
    parser.add_argument('--preset_model_type', type=str, default=None, help='set tokenizer, model_dir and model_type in preset modes. albert, deberta and electra, roberta.')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--predict_dir', type=str, default='/workspace/data/yicxu/csqa/jslin_model/prediction/', help='directory of prediction files.')

    # adv training parameters
    parser.add_argument('--adv_train', action='store_true')
    parser.add_argument('--adv_sift', action='store_true', help='use sift implementation of VAT.')
    parser.add_argument('--adv_norm_level', default=0, type=int)
    parser.add_argument('--adv_p_norm', default='inf', type=str)
    parser.add_argument('--adv_alpha', default=1, type=float)
    parser.add_argument('--adv_k', default=1, type=int)
    parser.add_argument('--adv_step_size', default=1e-5, type=float)
    parser.add_argument('--adv_noise_var', default=1e-5, type=float)
    parser.add_argument('--adv_epsilon', default=1e-6, type=float)
    parser.add_argument('--grad_adv_loss', default='symmetric-kl', type=str, help='loss for computing gradient in VAT. only useful for sift.')
    parser.add_argument('--adv_loss', default='SymKlCriterion', type=str)
    # Data parameters
    parser.add_argument('--append_answer_text', type=int, default=0, help='append answer text to the question.')
    parser.add_argument('--append_descr', type=int, default=0, help='append wiktionary description.')
    parser.add_argument('--append_retrieval', type=int, default=0, help='number of retrieval text to add, for obqa and csqa.')
    parser.add_argument('--append_triples', dest='no_triples', action='store_false', help='appending triples to the input.')
    parser.add_argument('--freq_rel', type=int, default=0, help='use most frequent relation. 0: None, 1: mask the softmax prediction based on most frequent relation.')
    parser.add_argument('--freq_threshold', type=int, default=3, help='threshold for masking.')

    parser.add_argument('--num_choices', type=int, default=5, help='number of choices per question.')
    parser.add_argument('--vary_segment_id', action='store_true', help='vary segment id for question+context.')

    # Other parameters
    parser.add_argument('--print_step', type=int, default=100, help='evaluate every this number of training steps.')
    parser.add_argument('--print_loss_step', type=int, default=None, help='print loss every this number of training steps.')
    parser.add_argument('--print_number_per_epoch', type=int, default=None, help='evluate this number of times per epoch. If given, will override print_step.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--predict_dev',action='store_true', help='predict results on dev.')
    parser.add_argument('--fp16', type=str, default=0, help='1=use pytorch amp')
    parser.add_argument('--test_mode', action='store_true', help='run on first several samples to test the pipeline.')
    parser.add_argument('--clear_output_folder', action='store_true', help='clear output folder (for test purposes).')
    parser.add_argument('--continue_train', action='store_true', help='find possible previous records and continue training.')
    parser.add_argument('--save_interval_step', type=int, default=None, help='save every this number of epochs. Model will join the last checkpoint.')
    parser.add_argument('--save_every', action='store_true', help='store every time the model is evaluated.')
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if args.ddp or args.deepspeed:
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        if args.deepspeed:
            deepspeed.init_distributed()
        else:
            dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
        )
    args.fp16 = int(args.fp16)
    args.total_batch_size = args.batch_size * args.num_choices # total number of texts in a batch
    print(f'batch size: {args.batch_size}, total_batch_size: {args.total_batch_size}')
    if args.preset_model_type=='albert':
        args.bert_model_dir = 'albert-xxlarge-v2'
        args.bert_vocab_dir = 'albert-xxlarge-v2'
        args.model_type = 'albert'
    elif args.preset_model_type == 'deberta':
        args.bert_model_dir = 'microsoft/deberta-xlarge-mnli'
        args.bert_vocab_dir = 'microsoft/deberta-xlarge-mnli'
        args.model_type = 'deberta'
    elif args.preset_model_type == 'electra':
        args.bert_model_dir = 'google/electra-large-discriminator'
        args.bert_vocab_dir = 'google/electra-large-discriminator'
        args.model_type = 'electra'       
    elif args.preset_model_type == 'roberta':
        args.bert_model_dir = 'roberta-large'
        args.bert_vocab_dir = 'roberta-large'
        args.model_type = 'roberta' 
    elif args.preset_model_type == 'electra-base':
        args.bert_model_dir = 'google/electra-base-discriminator'
        args.bert_vocab_dir = 'google/electra-base-discriminator'
        args.model_type = 'electra'       
    elif args.preset_model_type in {'debertav2', 'debertav2-xxlarge'}:
        args.bert_model_dir = 'microsoft/deberta-v2-xxlarge'
        args.bert_vocab_dir = 'microsoft/deberta-v2-xxlarge'
        args.model_type = 'debertav2'       
    elif args.preset_model_type in {'debertav2-mnli', 'debertav2-xxlarge-mnli'}:
        args.bert_model_dir = 'microsoft/deberta-v2-xxlarge-mnli'
        args.bert_vocab_dir = 'microsoft/deberta-v2-xxlarge-mnli'
        args.model_type = 'debertav2'   
    elif args.preset_model_type == 'debertav2-xlarge':
        args.bert_model_dir = 'microsoft/deberta-v2-xlarge'
        args.bert_vocab_dir = 'microsoft/deberta-v2-xlarge'
        args.model_type = 'debertav2'          
    elif args.preset_model_type == 'debertav2-xlarge-mnli':
        args.bert_model_dir = 'microsoft/deberta-v2-xlarge-mnli'
        args.bert_vocab_dir = 'microsoft/deberta-v2-xlarge-mnli'
        args.model_type = 'debertav2'      
    elif args.preset_model_type == 'debertav3':
        args.bert_model_dir = 'microsoft/deberta-v3-large'
        args.bert_vocab_dir = 'microsoft/deberta-v3-large'
        args.model_type = 'debertav2'               

    if args.deepspeed:
        if args.deepspeed_config is None:
            args.deepspeed_config = args.model_type
        args.deepspeed_config = f'ds_configs/{args.deepspeed_config}_ds_config.json'     
        ds_config = json.load(open(args.deepspeed_config))
        ds_config['micro_batch_per_gpu'] = args.batch_size
        ds_config['train_batch_size'] = args.batch_size * dist.get_world_size()
        args.deepspeed_config = ds_config
        
    test_name = 'dev_data.json' if args.predict_dev else 'test_data.json'
    if args.train_file_name is None:
        args.train_file_name = os.path.join(os.environ['DATA_DIR'], args.data_version, 'train_data.json')
        args.devlp_file_name = os.path.join(os.environ['DATA_DIR'], args.data_version, 'dev_data.json')
        args.trial_file_name = os.path.join(os.environ['DATA_DIR'], args.data_version, test_name)
        if args.test_mode:
            print(args.train_file_name)

    if args.mission == 'output':
        pred_folder = 'dev' if args.predict_dev else 'test'
        if args.predict_dir == 'AMLT_OUTPUT':
            args.predict_dir = os.environ['OUTPUT_DIR']
        args.predict_dir = os.path.join(args.predict_dir, pred_folder)
        Path(args.predict_dir).mkdir(exist_ok=True, parents=True)
    if args.pred_file_name is not None:
        args.pred_file_name = os.path.join(args.predict_dir, args.pred_file_name)
        print('output to:', args.pred_file_name)
    if args.output_model_dir is None:
        args.output_model_dir = os.environ['OUTPUT_DIR']
        if args.test_mode:
            print('output model dir:', args.output_model_dir)
        if os.path.exists(args.output_model_dir) and args.clear_output_folder and args.local_rank == 0:
            print('clearing output folder.')
            shutil.rmtree(args.output_model_dir)
    if args.ddp:
        dist.barrier()

    record_fn = os.path.join(args.output_model_dir, 'training_records.json')
    args.load_training_dir = None
    args.start_training_epoch = 0
    args.start_global_step = 0
    args.training_records = None
    args.start_tb_step = 0
    args.resume_training = False

    if args.continue_train and os.path.isfile(record_fn):
        training_records = json.load(open(record_fn))
        print('restarting from checkpoint.')
        used_name = training_records['current_used_last_name']
        training_info_fn = os.path.join(args.output_model_dir, used_name, 'training_info.json')
        assert os.path.isfile(training_info_fn)
        print('used_name:', used_name)
        training_info_fn = os.path.join(args.output_model_dir, used_name, 'training_info.json')
        if os.path.isfile(training_info_fn):
            training_info = json.load(open(training_info_fn))
            args.start_training_epoch = training_info['epoch']
            args.start_global_step = training_info['global_step']
            args.start_tb_step = training_info['tb_step']
            args.loading_used_name = used_name
            args.load_training_dir = os.path.join(args.output_model_dir, used_name)
            args.bert_model_dir = args.load_training_dir
            print(f'loading result from dir {args.load_training_dir}')
            args.training_records = training_records
            args.training_info = training_info
            args.resume_training = True

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
    from transformers import AlbertTokenizer, ElectraTokenizer, DebertaTokenizer
    from specific.io import load_data
    from pathlib import Path
    from specific.tensor import make_dataloader
    from model.model import Model
    from utils.common import mkdir_if_notexist
    from transformers import AutoTokenizer
    import torch.distributed as dist

    args = get_args()

    
    print("args.fp16 is {}".format(args.fp16))
    assert args.mission in ('train', 'output')
    # ------------------------------------------------#
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # ------------------------------------------------#

    # ------------------------------------------------#
    experiment = 'conceptnet'
    print('load_vocab', args.bert_vocab_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_vocab_dir)
    args.sep_word = tokenizer.sep_token    
    if args.mission == 'train':
        print('load_data', args.train_file_name)
        train_data = load_data(experiment, args.train_file_name, type='json', config=args, is_train=True)

        print('load_data', args.devlp_file_name)
        devlp_data = load_data(experiment, args.devlp_file_name, type='json', config=args)

        if args.test_mode:
            test_num = 26
            train_data = train_data[-test_num:]
            devlp_data = devlp_data[-test_num:]
    elif args.mission == 'output':
        dataset_name = 'csqa'
        print('load_data', args.trial_file_name)
        devlp_data = load_data(experiment, args.trial_file_name, type='json', config=args)
    print('get dir {}'.format(args.output_model_dir))
    Path(args.output_model_dir).mkdir(exist_ok=True, parents=True)

    log_file = time.strftime("%Y-%m-%d-%H-%M-%S.log", time.gmtime())
    fh = logging.FileHandler(os.path.join(args.output_model_dir, log_file))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # ------------------------------------------------#

    # ------------------------------------------------#
    print('make dataloader ...')
    this_seed = args.seed + 100
    if args.mission == 'train':
        train_dataloader = make_dataloader(
            experiment, train_data, tokenizer, total_batch_size=args.total_batch_size,
            drop_last=False, max_seq_length=args.max_seq_length, vary_segment_id=args.vary_segment_id, config=args, seed=this_seed)  # 52 + 3

        print('train_data %d ' % len(train_data))
        train_dataloader = DataLoaderSampler(train_dataloader, args.data_version)

    devlp_dataloader = make_dataloader(
            experiment, devlp_data, tokenizer, total_batch_size=args.total_batch_size,
            drop_last=False, max_seq_length=args.max_seq_length, shuffle=False, vary_segment_id=args.vary_segment_id, config=args, seed=this_seed, dev=True)
    devlp_dataloaders = {
        args.data_version: devlp_dataloader
    }
    print('devlp_data %d ' % len(devlp_data))
        
    # ------------------------------------------------#

    # -------------------- main ----------------------#
    if args.mission == 'train':
        srt = SelectReasonableText(args)
        srt.init(Model)
        srt.train(train_dataloader, devlp_dataloaders, save_last=False, save_every=args.save_every)

        srt = SelectReasonableText
    elif args.mission == 'output':
        srt = SelectReasonableText(args)
        srt.init(Model)
        dataset_name = args.data_version
        dataloader = devlp_dataloaders[dataset_name]
        idx, result, label, predict = srt.trial(dataloader)

        content = ''
        length = len(result)
        right = 0
        for i, item in enumerate(tqdm(result)):
            if predict[i] == label[i]:
                right += 1
            content += '{},{},{},{}\n' .format(idx[i][0], item, label[i], predict[i])

        res_data = {'idx': idx, 'result': result, 'label': label, 'predict': predict}
        logger.info("accuracy is {}".format(right/length))
        with open(args.pred_file_name, 'w', encoding='utf-8') as f:
            f.write(content)    
        with open(args.pred_file_name.replace('.csv', '.json'), 'w', encoding='utf-8') as f:
            json.dump(res_data, f)
        with open(args.pred_file_name.replace('.csv', '_summary.json'), 'w', encoding='utf-8') as f:
            summary_data = {'correct': right, 'total': length, 'config': vars(args)}
            json.dump(summary_data, f)
  
    end = time.time()
    logger.info("start is {}, end is {}".format(start, end))
    logger.info("Running time: %.2f seconds"%(end-start))
    if args.ddp:
        dist.destroy_process_group()    

