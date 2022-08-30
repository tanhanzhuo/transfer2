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
# from transformers.utils import logging
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
from tqdm import trange,tqdm
# from paddle.metric import Metric, Accuracy, Precision, Recall
# from paddlenlp.data import Stack, Tuple, Pad, Dict
# from paddlenlp.data.sampler import SamplerHelper
# from paddlenlp.transformers import LinearDecayWithWarmup
# from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

import torch.nn as nn
# import paddle.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaClassificationHead

CONVERT = {
    'stance':{'NONE':0,'FAVOR':1,'AGAINST':2},
    'hate': {'normal':0,'hatespeech':1,'offensive':2},
    'sem-18':{'0':0,'1':1},
    'sem-17':{'neutral':0,'positive':1,'negative':2},
    'imp-hate':{'not_hate':0,'implicit_hate':1,'negative':2},
    'sem19-task5-hate':{'0':0,'1':1},
    'sem19-task6-offen':{'NOT':0,'OFF':1},
    'sem22-task6-sarcasm':{'0':0,'1':1},
    'sem18-task1-affect':{'0':0,'1':1,'2':2,'3':3},
    'sem21-task7-humor':{'0':0,'1':1}
}

class RobertaForMulti(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self.post_init()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        return_dict = False
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        return logits


from paddlenlp.data import Stack, Tuple, Pad, Dict
class DataCollatorMulti():
    def __init__(self, tokenizer, ignore_label, batch_pad=None):
        self.batch_pad = batch_pad
        if batch_pad is None:
            self.batch_pad = lambda samples, fn=Dict({
                'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
                'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # segment
                'labels': Stack(dtype="int64"),  # label
            }): fn(samples)
        else:
            self.batch_pad = batch_pad

    def __call__(self, examples):
        examples = self.batch_pad(examples)
        examples = [torch.tensor(e) for e in examples]
        return examples


tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        # default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm',
        default='stance,sem-18,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm,sem18-task1-affect,sem21-task7-humor',
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
        "--method",
        default='token',
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
        default='1e-5,2e-5,3e-5',
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
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0.1, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        default=None,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default='1,10,100,1000,10000', type=str, help="random seed for initialization")
    parser.add_argument(
        "--shot", default='10,20,40,80,160,320,640,1280,full', type=str, help="random seed for initialization")

    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels = batch
        logits = model(input_ids.cuda(), segment_ids.cuda())

        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]

    f1_ma = f1_score(label_all, pred_all,average='macro')
    f1_mi = f1_score(label_all, pred_all, average='micro')
    f1_we = f1_score(label_all, pred_all, average='weighted')
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (f1_ma, f1_mi, f1_we))
    return f1_ma, f1_mi, f1_we

def convert_example(example, label2idx):
    if example.get('special_tokens_mask') is not None:
        example.pop('special_tokens_mask')
    example['labels'] = label2idx[example['labels']]
    return example  # ['input_ids'], example['token_type_ids'], label, prob


def do_train(args):
    # set_seed(args.seed)
    print(args)
    data_all = datasets.load_from_disk(args.input_dir)
    label2idx = CONVERT[args.task]
    trans_func = partial(
        convert_example,
        label2idx=label2idx)
    train_ds = data_all['train']
    train_ds = train_ds.map(trans_func)
    if len(args.shot) > 0:
        if args.shot is not 'full':
            sample_num = int(args.shot)
            train_ds = train_ds.shuffle()
            select_idx = []
            select_idx_dic = {}
            for val in label2idx.values():
                select_idx_dic[val] = 0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                if select_idx_dic[label_tmp] < sample_num:
                    select_idx.append(idx)
                    select_idx_dic[label_tmp] += 1
            np.random.shuffle(select_idx)
            train_ds = train_ds.select(select_idx)

    dev_ds = data_all['dev']
    dev_ds = dev_ds.map(trans_func)
    test_ds = data_all['test']
    test_ds = test_ds.map(trans_func)

    learning_rate = args.learning_rate.split(',')
    best_metric = [0, 0, 0]
    for lr in learning_rate:

        num_classes = len(label2idx.keys())
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMulti.from_pretrained(
            args.model_name_or_path, config=config).cuda()
        batchify_fn = DataCollatorMulti(tokenizer=tokenizer, ignore_label=-100)
        train_data_loader = DataLoader(
            train_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        dev_data_loader = DataLoader(
            dev_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        test_data_loader = DataLoader(
            test_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(lr))
        num_update_steps_per_epoch = len(train_data_loader)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_steps*args.max_train_steps),
            num_training_steps=args.max_train_steps,
        )

        loss_fct = nn.CrossEntropyLoss().cuda()

        print('start Training!!!')
        global_step = 0
        tic_train = time.time()

        for epoch in trange(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                input_ids, segment_ids, labels = batch
                logits = model(input_ids.cuda(), segment_ids.cuda())
                loss = loss_fct(logits, labels.cuda().view(-1))
                # print(step)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (epoch + 1) % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                    % (global_step, args.max_train_steps, epoch,
                       loss, args.logging_steps / (time.time() - tic_train),
                       args.seed,float(lr),args.input_dir))
                tic_train = time.time()
            if (epoch + 1) % args.save_steps == 0:
                tic_eval = time.time()
                cur_metric = evaluate(model, dev_data_loader)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric[0]:
                    model_best = copy.deepcopy(model).cpu()
                    best_metric = cur_metric
        del model
        torch.cuda.empty_cache()

    model = model_best.cuda()
    cur_metric = evaluate(model, test_data_loader)
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
    del model
    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # r_dir = '/work/test/finetune/continue/'
    for shot in args.shot.split(','):
        for task in args.task_name.split(','):
            for model_name in args.model_name_or_path.split(','):  # [r_dir+'bertweet/']:
                ave_metric = []
                for seed in args.seed.split(','):
                    set_seed(int(seed))
                    args_tmp = copy.deepcopy(args)
                    args_tmp.task = task
                    args_tmp.input_dir = args.input_dir + task + '/' + args.method
                    args_tmp.seed = int(seed)
                    args_tmp.shot = shot
                    args_tmp.model_name_or_path = model_name
                    ave_metric.append(do_train(args_tmp))
                ave_metric = np.array(ave_metric)
                print("*************************************************************************************")
                print('Task: %s, model: %s, shot: %s' % (task, model_name, shot))
                print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:, 0]) / 5,
                                                                    sum(ave_metric[:, 1]) / 5,
                                                                    sum(ave_metric[:, 2]) / 5))
                with open(args.results_name, 'a') as f_res:

                    f_res.write('Task: %s, model: %s, shot: %s\n' % (task, model_name, shot) )
                    f_res.write('aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (sum(ave_metric[:, 0]) / 5,
                                                                            sum(ave_metric[:, 1]) / 5,
                                                                            sum(ave_metric[:, 2]) / 5))
                    for tmp in range(5):
                        f_res.write('%.5f, %.5f, %.5f \n' % (ave_metric[tmp, 0],ave_metric[tmp, 1],ave_metric[tmp, 2]))

                    f_res.close()
