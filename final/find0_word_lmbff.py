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

def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

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
import json
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
    get_linear_schedule_with_warmup
)
from transformers.modeling_outputs import MaskedLMOutput
from accelerate import Accelerator
from tqdm import trange, tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import torch.nn as nn
# import paddle.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaLMHead
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer
from openprompt.prompts import ManualVerbalizer

from openprompt.plms import load_plm
from openprompt.prompts import ManualVerbalizer, ManualTemplate
from openprompt.prompts.prompt_generator import LMBFFTemplateGenerationTemplate
from openprompt.plms import load_plm
from openprompt.prompts.prompt_generator import T5TemplateGenerator
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.trainer import ClassificationRunner
from transformers import AdamW, get_linear_schedule_with_warmup
from openprompt.prompts.prompt_generator import RobertaVerbalizerGenerator, VerbalizerGenerator, TemplateGenerator
from transformers import RobertaForMaskedLM
from typing import List, Optional, Dict

CONVERT = {
    'eval-emotion': {'0': 0, '1': 1, '2': 2, '3': 3},
    'eval-hate': {'0': 0, '1': 1},
    'eval-irony': {'0': 0, '1': 1},
    'eval-offensive': {'0': 0, '1': 1},
    'eval-stance': {'0': 0, '1': 1, '2': 2},
    'stance': {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}

WORDS = {
    # 'stance':[["yes"], ["agree","like","favor"], ["dis","don't","not","hate"]],
    'eval-emotion': ["angerous", "joyful", "optimistic", "sad"],
    'eval-hate': ["neutral", "hateful"],
    'eval-irony': ["neutral", "ironic"],
    'eval-offensive': ["neutral", "offensive"],
    'eval-stance': ["neutral", "against", "favor"],
    'stance': ["neutral", "favor", "against"],
    'sem22-task6-sarcasm': ["neutral", "sarcastic"],
    'sem21-task7-humor': ["neutral", "humorous"]
}

TEMPLATE = {
    'eval-emotion': ' . It was {"mask"}. ',
    'eval-hate': ' . It was {"mask"}. ',
    'eval-irony': ' . It was {"mask"}. ',
    'eval-offensive': ' . It was {"mask"}. ',
    'eval-stance': ' . It was {"mask"}. ',
    'stance': ' . It was {"mask"}. ',
    'sem22-task6-sarcasm': ' . It was {"mask"}. ',
    'sem21-task7-humor': ' . It was {"mask"}. '
}


class BertweetVerbalizerGenerator(VerbalizerGenerator):
    def invalid_label_word(self, word: str):
        return ('@@' in word)
    def _eval_group(self, group):
        if len(set(group)) < len(group):
            return 0.0
        label_logits = self.probs_buffer[:,torch.tensor(group)]
        preds = torch.argmax(label_logits, axis=-1)
        # correct = torch.sum(preds == self.labels_buffer)
        # return (correct / len(self.labels_buffer)).item()
        return f1_score(self.labels_buffer.cpu().numpy(),preds.cpu().numpy(),average='macro')
    def post_process(self, word: str):
        return word.lstrip('Ġ')


class ManualTemplateWithoutParse(ManualTemplate):
    """The generated template from TemplateGenerator is a list of dict of parsed template_text. So no further parsing is needed."""

    def on_text_set(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm',
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
        default='',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=30,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--seed", default='0,1,2,3,4', type=str, help="random seed for initialization")
    parser.add_argument(
        "--generate_tmp", default=0, type=int, help="template generation")
    parser.add_argument(
        "--generate_word", default=1, type=int, help="label word generation")
    parser.add_argument(
        "--word", default=50, type=int, help="label word generation")
    parser.add_argument(
        "--shot", default='full', type=str, help="samples for generation")
    parser.add_argument(
        "--name", default='bertweet_words.txt', type=str, help="write name")
    parser.add_argument(
        "--pre_tmp", default=[],#['{"placeholder":"text_a"} It was {"mask"}. '],
        nargs='+', help="write name")
    parser.add_argument(
        "--pre_word", default=[],#["neutral,offensive"],
        nargs='+', help="write name")
    parser.add_argument(
        "--max_seq_length", default=64, type=int, help="write name")
    args = parser.parse_args()
    return args


def read_data(fileName):
    data_dic = []
    with open(fileName, 'r') as f:
        for line in f:
            data_dic.append(json.loads(line))
    return data_dic


@torch.no_grad()
def evaluate(model, data_loader, task='eval-emoji', write_result=''):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        # input_ids, segment_ids, labels = batch
        # logits = model(input_ids.cuda(), segment_ids.cuda())
        logits = model(batch.cuda())
        labels = batch['label']
        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
    if len(write_result) > 0:
        with open(write_result, 'a', encoding='utf-8') as f:
            f.write(task + '\n')
            for one in pred_all:
                f.write(str(one))
            f.write('\n')
    results = classification_report(label_all, pred_all, output_dict=True)

    if 'emoji' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Emotion (Macro f1)
    elif 'emotion' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Hate (Macro f1)
    elif 'hate' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Irony (Irony class f1)
    elif 'irony' in task:
        tweeteval_result = results['1']['f1-score']

        # Offensive (Macro f1)
    elif 'offensive' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Sentiment (Macro Recall)
    elif 'sentiment' in task:
        tweeteval_result = results['macro avg']['recall']

        # Stance (Macro F1 of 'favor' and 'against' classes)
    elif 'stance' in task:
        f1_against = results['1']['f1-score']
        f1_favor = results['2']['f1-score']
        tweeteval_result = (f1_against + f1_favor) / 2
    elif 'sarcasm' in task:
        tweeteval_result = results['1']['f1-score']
    elif 'humor' in task:
        tweeteval_result = results['1']['f1-score']

    # print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (tweeteval_result, tweeteval_result, tweeteval_result))
    return tweeteval_result


def train_epoch(model, train_dataloader, loss_func, optimizer):
    model.train()
    loss_all = []
    for step, inputs in enumerate(train_dataloader):
        inputs = inputs.cuda()
        logits = model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        loss_all.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    return np.mean(loss_all)


def fit(model, train_dataloader, val_dataloader, loss_func, optimizer, task):
    best_score = 0.0
    for epoch in range(10):
        train_loss = train_epoch(model, train_dataloader, loss_func, optimizer)
        if train_loss > 0.5:
            continue
        score = evaluate(model, val_dataloader, task)
        if score > best_score:
            best_score = score
        print(f"Epoch {epoch + 1}: Train loss={train_loss}, Eval score={score}")
    return best_score


def evaluate_tmp_word(tokenizer, template_text, verbalizer, args, dataset, plm):
    score_all = []
    for seed in args.seed.split(','):
        set_seed(int(seed))

        if isinstance(template_text, list):
            template = ManualTemplateWithoutParse(tokenizer, template_text)
        else:
            template = ManualTemplate(tokenizer, template_text)
        train_dataloader = PromptDataLoader(dataset['train'], template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=MLMTokenizerWrapper, shuffle=True, max_seq_length=args.max_seq_length,
                                            batch_size=args.batch_size)
        valid_dataloader = PromptDataLoader(dataset['dev'], template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=args.max_seq_length,
                                            batch_size=args.batch_size)

        model = PromptForClassification(copy.deepcopy(plm).cuda(), template, verbalizer)

        if 'sarcasm' in args.task:
            class_weights = [1,3.0]
            class_weights = torch.FloatTensor(class_weights).cuda()
            loss_func = nn.CrossEntropyLoss(weight=class_weights).cuda()
        else:
            loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        model = model.cuda()
        score = fit(model, train_dataloader, valid_dataloader, loss_func, optimizer, args.task)
        score_all.append(score)
        del model
        torch.cuda.empty_cache()
    return np.median(score_all)


def do_train(args):
    print(args)
    with open(args.name, 'a', encoding='utf-8') as f:
        f.write('******************************************\n')
        f.write(args.task + '\n')
    label2idx = CONVERT[args.task]
    dataset = {}
    SPLITS = ['train', 'dev', 'test']
    for split in SPLITS:
        dataset[split] = []
        data_all = read_data(args.input_dir + split + args.method + '.json')
        random.shuffle(data_all)
        for data in data_all:
            input_example = InputExample(text_a=data['text'], label=int(label2idx[data['labels']]))
            dataset[split].append(input_example)

    ##################load models

    if 'roberta' in args.model_name_or_path:
        plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
        # plm = plm.cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        plm = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)#.cuda()
        wrapped_tokenizer = MLMTokenizerWrapper(tokenizer=tokenizer, truncate_method="head", max_seq_length=args.max_seq_length)
    verbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx.keys()),
                                  label_words=WORDS[args.task])
    if len(args.pre_tmp) < 1:
        template_text = '{"placeholder":"text_a"}' + TEMPLATE[args.task]
    else:
        template_text = args.pre_tmp[0]
    template = ManualTemplate(tokenizer, text=template_text)

    ###########################verberlizer
    # load generation model for word generation
    if args.generate_word == 1:
        if len(label2idx.keys()) > 3:
            label_word_num_per_class = 50
        else:
            label_word_num_per_class = int( 100 * (3/len(label2idx.keys())) )
        if 'roberta' in args.model_name_or_path:
            verbalizer_generator = RobertaVerbalizerGenerator(model=copy.deepcopy(plm).cuda(), tokenizer=tokenizer, candidate_num=args.word,
                                                              label_word_num_per_class=label_word_num_per_class)
        else:
            verbalizer_generator = BertweetVerbalizerGenerator(model=copy.deepcopy(plm).cuda(), tokenizer=tokenizer, candidate_num=args.word,
                                                               label_word_num_per_class=label_word_num_per_class)
        if args.shot != 'full':
            sampler = FewShotSampler(num_examples_per_label=int(args.shot))
            dataset_gen = sampler(dataset['train'])
        else:
            dataset_gen = dataset['train']
        dataloader = PromptDataLoader(dataset_gen, template, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=MLMTokenizerWrapper,
                                      batch_size=args.batch_size, max_seq_length=args.max_seq_length)
        for data in dataloader:
            data = data.cuda()
            verbalizer_generator.register_buffer(data)
        label_words_list = verbalizer_generator.generate()
        label_words_list.insert(0,WORDS[args.task])
        verbalizer_generator.release_memory()
        del verbalizer_generator, dataloader, data
        torch.cuda.empty_cache()
    else:
        if len(args.pre_word) < 1:
            label_words_list = [WORDS[args.task]]
        else:
            label_words_list = [i.split(',') for i in args.pre_word]
    print('label word list:')
    print(label_words_list)

    # iterate over each candidate and select the best one
    best_metrics = 0.0
    if len(label_words_list) > 1:
        best_label_words = None
        for label_words in tqdm(label_words_list):
            label_uni = set(label_words)
            if len(label_uni) != len(label_words):
                continue
            current_verbalizer = copy.deepcopy(verbalizer)
            current_verbalizer.label_words = label_words
            score = evaluate_tmp_word(tokenizer, template_text, current_verbalizer, args, dataset, copy.deepcopy(plm))
            with open(args.name, 'a', encoding='utf-8') as f:
                f.write(' '.join(label_words) + '\n')
                f.write('{:.5f}'.format(score) + '\n')
            if score > best_metrics:
                best_metrics = score
                best_label_words = label_words
    else:
        best_label_words = label_words_list[0]
    # use the best verbalizer
    verbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx.keys()), label_words=best_label_words)
    print("final best label words:")
    print(best_label_words)

    with open(args.name, 'a', encoding='utf-8') as f:
        f.write(' '.join(best_label_words) + '\n')
        f.write('{:.5f}'.format(best_metrics) + '\n')



    del plm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    for task in args.task.split(','):
        args_tmp = copy.deepcopy(args)
        args_tmp.task = task.split('_')[0]
        args_tmp.input_dir = args.input_dir + task + '/'
        do_train(args_tmp)

