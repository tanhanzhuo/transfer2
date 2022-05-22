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
import torch
from functools import partial
import copy
import numpy as np
import datasets
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
        "--task_name",
        default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm',
        type=str,
        required=False,
        help="The name of the task to train selected in the list: ")
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
        default='../finetune/data/',
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
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--seed", default='1,10,100,1000,10000', type=str, help="random seed for initialization")
    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        labels = batch.pop('labels')
        outputs = model(**batch)
        preds = outputs.logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]

    f1_ma = f1_score(label_all, pred_all,average='macro')
    f1_mi = f1_score(label_all, pred_all, average='micro')
    f1_we = f1_score(label_all, pred_all, average='weighted')
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (f1_ma, f1_mi, f1_we))
    return f1_ma, f1_mi, f1_we

def read_label(data):
    label_name = set()
    for one in data:
        label_name.add(one['labels'])
    label2idx = {}
    label_name = sorted(list(label_name))
    for idx in range(0, len(label_name)):
        label2idx[label_name[idx]] = idx
    return label2idx


def convert_example(example, label2idx):
    # example.pop('attention_mask')
    example.pop('special_tokens_mask')
    example['labels'] = label2idx[example['labels']]
    return example  # ['input_ids'], example['token_type_ids'], label, prob

def do_train(args):
    # set_seed(args.seed)
    print(args)
    data_all = datasets.load_from_disk(args.input_dir)
    label2idx = read_label(data_all['train'])
    trans_func = partial(
        convert_example,
        label2idx=label2idx)
    test_ds = data_all['test']
    test_ds = test_ds.map(trans_func)

    accelerator = Accelerator()
    num_classes = len(label2idx.keys())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    test_data_loader = DataLoader(
        test_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
    )
    print('data ready!!!')

    model, test_data_loader = accelerator.prepare(model, test_data_loader)
    cur_metric = evaluate(model, test_data_loader)
    print('final')

    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    for task in args.task_name.split(','):
        for model_name in args.model_name_or_path.split(','):
            ave_metric = []
            for seed in args.seed.split(','):
                set_seed(int(seed))
                args_tmp = copy.deepcopy(args)
                args_tmp.input_dir = args.input_dir + task + '/token'
                args_tmp.seed = int(seed)
                args_tmp.model_name_or_path = model_name
                ave_metric.append(do_train(args_tmp))
            ave_metric = np.array(ave_metric)
            print("*************************************************************************************")
            print('Task: %s, model: %s' % (task, model_name))
            print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:, 0]) / 5,
                                                                sum(ave_metric[:, 1]) / 5,
                                                                sum(ave_metric[:, 2]) / 5))
            with open(args.results_name, 'a') as f_res:

                f_res.write('Task: %s, model: %s\n' % (task, model_name) )
                f_res.write('aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (sum(ave_metric[:, 0]) / 5,
                                                                        sum(ave_metric[:, 1]) / 5,
                                                                        sum(ave_metric[:, 2]) / 5))
                for tmp in range(5):
                    f_res.write('%.5f, %.5f, %.5f \n' % (ave_metric[tmp, 0],ave_metric[tmp, 1],ave_metric[tmp, 2]))

                f_res.close()
