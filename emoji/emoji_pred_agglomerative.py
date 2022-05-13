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
import datasets
# import paddle
# from paddle.io import DataLoader

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default='vinai/bertweet-base',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )
    parser.add_argument(
        "--token_name_or_path",
        default='vinai/bertweet-base',
        type=str,
        required=False,
    )
    parser.add_argument(
        "--input_dir",
        default='./data/',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_classes",
        default=20,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--output_dir",
        default='./model/',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--results_name",
        default='results_all.txt',
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
        default='1e-5',
        type=str,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=30,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
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
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        default=None,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="0",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")

    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        labels = batch['labels']
        # batch.pop('special_tokens_mask')
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
    rep = classification_report(label_all, pred_all,
                                digits=5, output_dict=True)
    f1 = rep['macro avg']['f1-score']
    precision = rep['macro avg']['precision']
    recall = rep['macro avg']['recall']
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (f1, precision, recall))
    return f1, precision, recall

def do_train(args):
    # set_seed(args.seed)
    accelerator = Accelerator()
    data_all = datasets.load_from_disk(args.input_dir)
    train_ds = data_all['train']
    dev_ds = data_all['dev']

    num_classes = args.num_classes
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    train_data_loader = DataLoader(
        train_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
    )
    dev_data_loader = DataLoader(
        dev_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
    )

    print('data ready!!!')
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(args.learning_rate))
    num_update_steps_per_epoch = len(train_data_loader)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_data_loader, dev_data_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, dev_data_loader, lr_scheduler
    )

    print('start Training!!!')
    global_step = 0
    tic_train = time.time()
    best_metric = [0, 0, 0]
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            # batch.pop('special_tokens_mask')
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if (global_step + 1) % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s"
                    % (global_step, args.max_train_steps, epoch,
                       loss.item(), args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            if (global_step + 1) % args.save_steps == 0:
                tic_eval = time.time()

                cur_metric = evaluate(model, dev_data_loader)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric[0]:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(args.output_dir)
                    best_metric = cur_metric
                    del unwrapped_model
                    torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(args.output_dir + str(epoch))
        best_metric = cur_metric
        del unwrapped_model
        torch.cuda.empty_cache()
    del model#, optimizer, logits, logits_seq, loss, loss_seq, loss_all, accelerator
    torch.cuda.empty_cache()

    return cur_metric


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
