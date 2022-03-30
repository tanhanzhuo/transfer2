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

import datasets
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.data.sampler import SamplerHelper
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report

from transformers import AutoTokenizer

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import RobertaPretrainedModel
class RobertaForMulti(RobertaPretrainedModel):

    def __init__(self, roberta, num_classes=2, seq_classes=2, dropout=None):
        super(RobertaForMulti, self).__init__()
        self.num_classes = num_classes
        self.roberta = roberta  # allow roberta to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.roberta.config["hidden_size"],
                                    num_classes)

        self.dropout_seq = nn.Dropout(dropout if dropout is not None else
                                  self.roberta.config["hidden_dropout_prob"])
        self.classifier_seq = nn.Linear(self.roberta.config["hidden_size"],
                                    seq_classes)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        sequence_output, pooled_output = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        sequence_output = self.dropout_seq(sequence_output)
        logits_seq = self.classifier_seq(sequence_output)

        return logits, logits_seq

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
def parse_args():
    parser = argparse.ArgumentParser()

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
        "--ratio", default=10, type=int, help="ratio for loss")
    parser.add_argument(
        "--ratio2", default=10, type=int, help="ratio for loss")
    parser.add_argument(
        "--device",
        default="gpu:1",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
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
def evaluate(model, data_loader):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels, labels_seq = batch
        logits, logits_seq = model(input_ids, segment_ids)

        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
    names = ['neu', 'neg', 'pos']
    rep = classification_report(label_all, pred_all,
                            target_names=names, digits=5, output_dict=True)
    aveRec =  rep['macro avg']['recall']
    f1PN = (rep['pos']['f1-score'] + rep['neg']['f1-score'])/2
    acc = rep['accuracy']
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % ( aveRec, f1PN, acc))
    model.train()
    return aveRec, f1PN, acc

@paddle.no_grad()
def evaluate_18(model, data_loader):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels, labels_seq = batch
        logits, logits_seq = model(input_ids, segment_ids)

        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
    names = ['not', 'irony']
    rep = classification_report(label_all, pred_all,
                            target_names=names, digits=5, output_dict=True)
    f1_pos =  rep['irony']['f1-score']
    f1PN = (rep['not']['f1-score'] + rep['irony']['f1-score'])/2
    acc = rep['accuracy']
    print("f1_pos:%.5f, f1PN:%.5f, acc: %.5f " % (f1_pos, f1PN, acc))
    model.train()
    return f1_pos, f1PN, acc

def read_label(data):
    label_name = set()
    for one in data:
        label_name.add(one['label'])
    label2idx = {}
    label_name = list(label_name)
    for idx in range(0, len(label_name)):
        label2idx[label_name[idx]] = idx
    return label2idx


def convert_example(example, label2idx):
    example.pop('special_tokens_mask')
    example.pop('attention_mask')
    example['label'] = label2idx[example['label']] 
    example['prob_map'] = [int(0) if i<0.5 else int(1) for i in example['prob_map']]

    return example#['input_ids'], example['token_type_ids'], label, prob


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args)
    # tokenizer = AutoTokenizer.from_pretrained(args.token_name_or_path, normalization=True)
    
    data_all = datasets.load_from_disk(args.input_dir)
    if 'sem-18' in args.input_dir:
        label2idx = {'0': 0, '1': 1}
    else:
        # label2idx = read_label(data_all['train'])
        label2idx = {'neutral': 0, 'negative': 1, 'positive':2}
    train_ds = data_all['train']
    trans_func = partial(
        convert_example,
        label2idx = label2idx)
    train_ds = train_ds.map(trans_func)
    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True)

    ignore_label = -100
    batchify_fn = lambda samples, fn=Dict({
        'input_ids':Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
        'token_type_ids':Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # segment
        'label':Stack(dtype="int64"),  # label
        'prob_map':Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)


    train_data_loader = DataLoader(
        dataset=train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    dev_ds = data_all['dev']
    dev_ds = dev_ds.map(trans_func)
    dev_batch_sampler = paddle.io.BatchSampler(
        dev_ds, batch_size=args.batch_size, shuffle=False)
    dev_data_loader = DataLoader(
        dataset=dev_ds,
        batch_sampler=dev_batch_sampler,
        collate_fn=batchify_fn,
        num_workers=0,
        return_list=True)

    num_classes = len(label2idx.keys())
    model = RobertaForMulti.from_pretrained(
        args.model_name_or_path, num_classes=num_classes, seq_classes=2)

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

    loss_fct = paddle.nn.loss.CrossEntropyLoss()

    weight = [args.ratio2/5.0,2.0-args.ratio2/5.0]
    weight = paddle.to_tensor(np.array(weight), dtype="float32")
    loss_fct_seq = paddle.nn.loss.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
    # loss_fct_seq = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    if args.use_amp:
        scaler = paddle.amp.GradScaler(init_loss_scaling=args.scale_loss)

    global_step = 0
    tic_train = time.time()
    best_metric = [0,0,0]
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1

            input_ids, segment_ids, labels, labels_seq = batch
            with paddle.amp.auto_cast(
                    args.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits, logits_seq = model(input_ids, segment_ids)

                loss = loss_fct(logits, labels)
                loss_seq = loss_fct_seq(logits_seq, labels_seq)
                loss_all = loss*args.ratio/10.0 + loss_seq*(10-args.ratio)/10.0
            if args.use_amp:
                scaler.scale(loss_all).backward()
                scaler.minimize(optimizer, loss_all)
            else:
                loss_all.backward()
                optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        if (epoch+1) % args.logging_steps == 0:
            print(
                "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
                % (global_step, num_training_steps, epoch, step,
                    paddle.distributed.get_rank(), loss_all, optimizer.get_lr(),
                    args.logging_steps / (time.time() - tic_train)))
            tic_train = time.time()
        if (epoch+1) % args.save_steps == 0:
            tic_eval = time.time()
            if 'sem-18' in args.input_dir:
                cur_metric = evaluate_18(model, dev_data_loader)
            else:
                cur_metric = evaluate(model, dev_data_loader)
            print("eval done total : %s s" % (time.time() - tic_eval))
            if cur_metric[0] > best_metric[0]:
                if paddle.distributed.get_rank() == 0:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                best_metric = cur_metric

    model = RobertaForMulti.from_pretrained(
    args.output_dir, num_classes=len(label2idx.keys()), seq_classes=2)   
    test_ds = data_all['test']
    test_ds = test_ds.map(trans_func)

    test_data_loader = DataLoader(
        dataset=test_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        return_list=True)
    if 'sem-18' in args.input_dir:
        cur_metric = evaluate_18(model, test_data_loader)
    else:
        cur_metric = evaluate(model, test_data_loader)
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
    return cur_metric


if __name__ == "__main__":
    args = parse_args()
    r_dir = '/work/test/finetune/continue/'
    for task in ['sem-18']:
        for model_name in [r_dir+'bertweet/']:
            for ratio in range(10,0,-2):
                for ratio2 in range(10,-2,-2):
                    ave_metric = []
                    for seed in [1, 10, 100, 1000, 10000, 2,20,200,2000,20000]:
                        args_tmp = copy.deepcopy(args)
                        args_tmp.input_dir = '/work/test/finetune/data/' + task + '/prob'
                        args_tmp.output_dir = '/work/test/finetune/model/' + task + '/'
                        args_tmp.seed = seed
                        args_tmp.model_name_or_path = model_name
                        args_tmp.token_name_or_path = 'vinai/bertweet-base'
                        args_tmp.ratio = ratio
                        args_tmp.ratio2 = ratio2
                        ave_metric.append(do_train(args_tmp))
                    ave_metric = np.array(ave_metric)
                    print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:,0])/10, 
                                                                        sum(ave_metric[:,1])/10, 
                                                                        sum(ave_metric[:,2])/10) )
                    with open('results_sem_continue_bertweet.txt', 'a') as f_res:
                        f_res.write('loss ratio: '+str(ratio) + ' weight ratio: '+str(ratio2)+ '\n')
                        f_res.write('Task: %s, aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (task, sum(ave_metric[:,0])/10, 
                                                                                    sum(ave_metric[:,1])/10, 
                                                                                    sum(ave_metric[:,2])/10) )
                        f_res.close()