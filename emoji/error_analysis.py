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

# from paddle.metric import Metric, Accuracy, Precision, Recall
# from paddlenlp.data import Stack, Tuple, Pad, Dict
# from paddlenlp.data.sampler import SamplerHelper
# from paddlenlp.transformers import LinearDecayWithWarmup
# from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

import torch.nn as nn
# import paddle.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaClassificationHead


class RobertaForMulti(RobertaPreTrainedModel):

    def __init__(self, config, config2):
        super().__init__(config, config2)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.classifier_emoji = RobertaClassificationHead(config2)

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
        logits_emoji = self.classifier_emoji(sequence_output)
        return logits, logits_emoji


from paddlenlp.data import Stack, Tuple, Pad, Dict
class DataCollatorMulti():
    def __init__(self, tokenizer, ignore_label, batch_pad=None):
        self.batch_pad = batch_pad
        if batch_pad is None:
            self.batch_pad = lambda samples, fn=Dict({
                'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),  # input
                'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),  # segment
                'labels': Stack(dtype="int64"),  # label
                'emoji': Stack(dtype="int64")  # label
            }): fn(samples)
        else:
            self.batch_pad = batch_pad

    def __call__(self, examples):
        examples = self.batch_pad(examples)
        examples = [torch.tensor(e) for e in examples]
        return examples


tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
emoji_top = ['????', '????', '????', '????', '????', '????', '????', '????', '????', '????', '????', '???', '???', '????', '????', '????', '??????',
                 '????', '????', '????']

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task_name",
        default='stance,stance/face_masks_sp,stance/fauci_sp,stance/school_closures_sp,stance/stay_at_home_orders_sp,hate,sem-17,sem-18,wtwt,wtwt/CVS_AET,wtwt/CI_ESRX,wtwt/ANTM_CI,wtwt/AET_HUM,wtwt/FOXA_DIS',
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
        "--seed", default='1,10,100,1000,10000', type=str, help="random seed for initialization")
    parser.add_argument(
        "--ratio", default='10', type=str, help="ratio for loss")
    parser.add_argument(
        "--ratio2", default='5', type=str, help="ratio for loss")
    parser.add_argument(
        "--device",
        default="0",
        type=str,
        help="The device to select to train the model, is must be cpu/gpu/xpu.")

    args = parser.parse_args()
    return args

import json
def write_json(fileName, data):
    with open(fileName + '.json', 'a', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

@torch.no_grad()
def evaluate(model, data_loader, output_name=None):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels, labels_seq = batch
        logits, logits_seq = model(input_ids, segment_ids)

        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
        if output_name:
            preds_seq = logits_seq.argmax(axis=1).cpu().numpy()
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            labels_seq = labels_seq.cpu().numpy()
            data = []
            for idx in range(len(labels)):
                txt = {#'text':tokenizer.decode(input_ids[idx].cpu()).replace(' <pad>',''),
                        'pred':str(preds[idx]),
                        'label':str(labels[idx]),
                        'emoji_pred':emoji_top[preds_seq[idx]],
                        'emoji_lab': emoji_top[labels_seq[idx]]
                       }
                data.append(txt)
            write_json(output_name, data)
    rep = classification_report(label_all, pred_all,
                                digits=5, output_dict=True)
    f1 = rep['macro avg']['f1-score']
    acc = rep['accuracy']
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (f1, acc, acc))
    return f1, acc, acc

@torch.no_grad()
def evaluate_17(model, data_loader, output_name=None):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels, labels_seq = batch
        logits, logits_seq = model(input_ids, segment_ids)

        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
        if output_name:
            preds_seq = logits_seq.argmax(axis=1).cpu().numpy()
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            labels_seq = labels_seq.cpu().numpy()
            data = []
            for idx in range(len(labels)):
                txt = {#'text':tokenizer.decode(input_ids[idx].cpu()).replace(' <pad>',''),
                        'pred':str(preds[idx]),
                        'label':str(labels[idx]),
                        'emoji_pred':emoji_top[preds_seq[idx]],
                        'emoji_lab': emoji_top[labels_seq[idx]]
                       }
                data.append(txt)
            write_json(output_name, data)
    names = ['neu', 'neg', 'pos']
    rep = classification_report(label_all, pred_all,
                                target_names=names, digits=5, output_dict=True)
    aveRec = rep['macro avg']['recall']
    f1PN = (rep['pos']['f1-score'] + rep['neg']['f1-score']) / 2
    acc = rep['accuracy']
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (aveRec, f1PN, acc))
    return aveRec, f1PN, acc


@torch.no_grad()
def evaluate_18(model, data_loader, output_name=None):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, segment_ids, labels, labels_seq = batch
        logits, logits_seq = model(input_ids, segment_ids)

        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
        if output_name:
            preds_seq = logits_seq.argmax(axis=1).cpu().numpy()
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            labels_seq = labels_seq.cpu().numpy()
            data = []
            for idx in range(len(labels)):
                txt = {#'text':tokenizer.decode(input_ids[idx].cpu()).replace(' <pad>',''),
                        'pred':str(preds[idx]),
                        'label':str(labels[idx]),
                        'emoji_pred':emoji_top[preds_seq[idx]],
                        'emoji_lab': emoji_top[labels_seq[idx]]
                       }
                data.append(txt)
            write_json(output_name, data)
    names = ['not', 'irony']
    rep = classification_report(label_all, pred_all,
                                target_names=names, digits=5, output_dict=True)
    f1_pos = rep['irony']['f1-score']
    f1PN = (rep['not']['f1-score'] + rep['irony']['f1-score']) / 2
    acc = rep['accuracy']
    print("f1_pos:%.5f, f1PN:%.5f, acc: %.5f " % (f1_pos, f1PN, acc))
    return f1_pos, f1PN, acc


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
    example.pop('attention_mask')
    example['labels'] = label2idx[example['labels']]
    return example  # ['input_ids'], example['token_type_ids'], label, prob


def do_train(args):
    # set_seed(args.seed)
    print(args)
    data_all = datasets.load_from_disk(args.input_dir)
    if 'sem-18' in args.input_dir:
        label2idx = {'0': 0, '1': 1}
    elif'sem-17' in args.input_dir:
        label2idx = {'neutral': 0, 'negative': 1, 'positive': 2}
    else:
        label2idx = read_label(data_all['train'])
    trans_func = partial(
        convert_example,
        label2idx=label2idx)
    train_ds = data_all['train']
    train_ds = train_ds.map(trans_func)
    dev_ds = data_all['dev']
    dev_ds = dev_ds.map(trans_func)
    test_ds = data_all['test']
    test_ds = test_ds.map(trans_func)

    learning_rate = args.learning_rate.split(',')
    best_metric = [0, 0, 0]
    for lr in learning_rate:
        accelerator = Accelerator()
        num_classes = len(label2idx.keys())
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
        config2 = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=20)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMulti.from_pretrained(
            args.model_name_or_path, config=config, config2=config2)

        batchify_fn = DataCollatorMulti(tokenizer=tokenizer, ignore_label=-100)
        train_data_loader = DataLoader(
            train_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        dev_data_loader = DataLoader(
            dev_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        test_data_loader = DataLoader(
            test_ds, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
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
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        model, optimizer, train_data_loader, dev_data_loader, test_data_loader = accelerator.prepare(
            model, optimizer, train_data_loader, dev_data_loader, test_data_loader
        )

        loss_fct = nn.CrossEntropyLoss().cuda()
        loss_fct_emoji = nn.CrossEntropyLoss().cuda()

        print('start Training!!!')
        global_step = 0
        tic_train = time.time()

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                input_ids, segment_ids, labels, labels_emoji = batch
                logits, logits_emoji = model(input_ids, segment_ids)
                loss = loss_fct(logits, labels.view(-1))
                loss_emoji = loss_fct_emoji(logits_emoji, labels_emoji.view(-1))
                loss_all = loss * int(args.ratio) / 10.0 + loss_emoji * (10 - int(args.ratio)) / 10.0
                # print(step)
                accelerator.backward(loss_all)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (epoch + 1) % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                    % (global_step, args.max_train_steps, epoch,
                       loss_all, args.logging_steps / (time.time() - tic_train),
                       args.seed,float(lr),args.input_dir))
                tic_train = time.time()
            if (epoch + 1) % args.save_steps == 0:
                tic_eval = time.time()
                if 'sem-18' in args.input_dir:
                    cur_metric = evaluate_18(model, dev_data_loader)
                elif 'sem-17' in args.input_dir:
                    cur_metric = evaluate_17(model, dev_data_loader)
                else:
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
        del model#, optimizer, logits, logits_seq, loss, loss_seq, loss_all, accelerator
        torch.cuda.empty_cache()

    model = RobertaForMulti.from_pretrained(
        args.output_dir, config=config, config2=config2)
    model = accelerator.prepare(model)
    if 'sem-18' in args.input_dir:
        cur_metric = evaluate_18(model, test_data_loader, output_name=args.output_dir.split('/')[-2]+
                                                                    '_seed_'+str(args.seed)+
                                                                    '_ratio_'+args.ratio)
    elif 'sem-18' in args.input_dir:
        cur_metric = evaluate_17(model, test_data_loader, output_name=args.output_dir.split('/')[-2]+
                                                                    '_seed_'+str(args.seed)+
                                                                    '_ratio_'+args.ratio)
    else:
        cur_metric = evaluate(model, test_data_loader, output_name=args.output_dir.split('/')[-2]+
                                                                    '_seed_'+str(args.seed)+
                                                                    '_ratio_'+args.ratio)
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
    del model
    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # r_dir = '/work/test/finetune/continue/'
    for task in args.task_name.split(','):
        for model_name in args.model_name_or_path.split(','):  # [r_dir+'bertweet/']:
            for ratio in args.ratio.split(','):#range(10, 0, -2):
                for ratio2 in args.ratio2.split(','):#range(10, -2, -2):
                    ave_metric = []
                    for seed in args.seed.split(','):
                        set_seed(int(seed))
                        args_tmp = copy.deepcopy(args)
                        args_tmp.input_dir = args.input_dir + task + '/emoji'
                        args_tmp.output_dir = args.output_dir + task + '/'
                        args_tmp.seed = int(seed)
                        args_tmp.model_name_or_path = model_name
                        args_tmp.ratio = ratio
                        args_tmp.ratio2 = ratio2
                        ave_metric.append(do_train(args_tmp))
                    ave_metric = np.array(ave_metric)
                    print("*************************************************************************************")
                    print('Task: %s, model: %s, loss ration: %s, weight ratio: %s' % (task, model_name, ratio, ratio2))
                    print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:, 0]) / 5,
                                                                        sum(ave_metric[:, 1]) / 5,
                                                                        sum(ave_metric[:, 2]) / 5))
                    with open(args.results_name, 'a') as f_res:

                        f_res.write('Task: %s, model: %s, loss ration: %s, weight ratio: %s\n' % (task, model_name, ratio, ratio2 ) )
                        f_res.write('aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (sum(ave_metric[:, 0]) / 5,
                                                                                sum(ave_metric[:, 1]) / 5,
                                                                                sum(ave_metric[:, 2]) / 5))
                        for tmp in range(5):
                            f_res.write('%.5f, %.5f, %.5f \n' % (ave_metric[tmp, 0],ave_metric[tmp, 1],ave_metric[tmp, 2]))

                        f_res.close()
