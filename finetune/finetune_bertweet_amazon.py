# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import random
import time
import math
import distutils.util
from functools import partial
import copy
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report

from transformers import AutoTokenizer
from modeling import RobertaForSequenceClassification

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="gpu:0",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    # Required parameters
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=False,
        help="The name of the task to train selected in the list: ")
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        required=False,
        help="Model type selected in the list: " )
    parser.add_argument(
        "--model_name_or_path",
        default='bert-base-cased',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
        )
    parser.add_argument(
        "--token_name_or_path",
        default='bert-base-cased',
        type=str,
        required=False,
        )
    parser.add_argument(
        "--input_dir",
        default='./tmp/',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_dir",
        default='./tmp/',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=30,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=3,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps. If > 0: Override warmup_proportion"
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Linear warmup proportion over total steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-6,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")

    parser.add_argument(
        "--use_amp",
        type=distutils.util.strtobool,
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2**15,
        help="The value of scale_loss for fp16.")
    args = parser.parse_args()
    return args


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


@paddle.no_grad()
def evaluate(model, loss_fct, data_loader):
    model.eval()
    loss_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = loss_fct(logits, labels).item()
        loss_all.append(loss)

    print("MSE:%.5f " % ( np.mean(loss_all)))
    model.train()
    return np.mean(loss_all)

def read_data(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            yield {'sentence':line.split('\t')[1], 'label':float(line.split('\t')[0])}
        f.close()

def convert_example(example, tokenizer, max_seq_length=128):
        label = [example['label']]
        example = tokenizer(example['sentence'])#########, max_seq_len=max_seq_length)

        return example['input_ids'], example['token_type_ids'], label


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(args.token_name_or_path, normalization=True)

    train_ds = load_dataset(read_data, fileName=args.input_dir + 'train', lazy=False)
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func, lazy=True)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Stack(dtype="float32")  # label
    ): fn(samples)
    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_ds = load_dataset(read_data, fileName=args.input_dir + 'dev', lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=True)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=1)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    num_training_steps = args.max_steps if args.max_steps > 0 else (
        len(train_data_loader) * args.num_train_epochs)
    warmup = args.warmup_steps if args.warmup_steps > 0 else args.warmup_proportion

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         warmup)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    loss_fct = paddle.nn.loss.MSELoss()

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    best_metric = 1
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1

            input_ids, segment_ids, labels = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = model(input_ids, segment_ids)
                loss = loss_fct(logits, labels)
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.minimize(optimizer, loss)
            else:
                loss.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        if (epoch+1) % args.logging_steps == 0:
            print(
                "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                % (global_step, num_training_steps, epoch, step,
                    paddle.distributed.get_rank(), loss, optimizer.get_lr(),
                    args.logging_steps / (time.time() - tic_train)))
            tic_train = time.time()
        if (epoch+1) % args.save_steps == 0:
            tic_eval = time.time()
            
            cur_metric = evaluate(model, loss_fct, dev_data_loader)
            print("eval done total : %s s" % (time.time() - tic_eval))
            if cur_metric < best_metric:
                if paddle.distributed.get_rank() == 0:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                best_metric = cur_metric

    model = RobertaForSequenceClassification.from_pretrained(
    args.output_dir, num_classes=1)   
    test_ds = load_dataset(read_data, fileName=args.input_dir + 'test', lazy=False)
    test_ds = test_ds.map(trans_func)

    test_data_loader = DataLoader(
        dataset=test_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        return_list=True)
    
    cur_metric = evaluate(model, loss_fct, test_data_loader)
    print("dev MSE:%.5f " % (best_metric))
    print("test MSE:%.5f " % (cur_metric))
    return cur_metric


if __name__ == "__main__":
    args = parse_args()
    r_dir = '/work/test/finetune_newdata/convert_model/pd/'
    for task in ['amazon-review-helpful']:
    #     for model_name in [r_dir+'600000/', r_dir+'720000/', r_dir+'840000/', r_dir+'960000/', r_dir+'1080000/', r_dir+'1200000/', r_dir+'1320000/']:
        # for model_name in [r_dir+'bertweet/', r_dir+'400000/', r_dir+'800000/']:
        for model_name in [r_dir+'200000/', r_dir+'600000/', r_dir+'2200000/', r_dir+'2400000/']:
            ave_metric = []
            for seed in [1, 10, 100, 1000, 10000]:
                args_tmp = copy.deepcopy(args)
                args_tmp.input_dir = '/work/test/finetune_newdata/data/' + task + '/'
                args_tmp.output_dir = '/work/test/finetune_newdata/model/' + task + '/'
                args_tmp.seed = seed
                args_tmp.model_name_or_path = model_name
                args_tmp.token_name_or_path = 'vinai/bertweet-base'
                ave_metric.append(do_train(args_tmp))
            ave_metric = np.array(ave_metric)
            print('final average test ' % (sum(ave_metric)/5 ) )
            with open('results_amazon.txt', 'a') as f_res:
                f_res.write(model_name + '\n')
                f_res.write('Task: %s, MSE:%.5f \n' % (task, sum(ave_metric)/5) )
                f_res.close()