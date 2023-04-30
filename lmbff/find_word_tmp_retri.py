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
    'eval-stance': ["neutral", "against", "favor"],
    'eval-emotion': ["disgusting", "magical", "love", "sad"],
    'eval-irony': ["hard", "unexpected"],
    'eval-offensive': ["neutral", "offensive"],
    'eval-hate': ["interesting", "disgusting"],
    'sem21-task7-humor': ["real", "funny"],
    'sem22-task6-sarcasm': ["neutral", "sarcastic"]
}

TEMPLATE = {
    'eval-stance': ' D #SemSTD is {"mask"}.',
    'eval-emotion': ' . {"mask"}.',
    'eval-irony': ' . {"mask"}.',
    'eval-offensive': ' . This is so {"mask"}.',
    'eval-hate': ' . It was {"mask"}. ',
    'sem21-task7-humor': ' . It was {"mask"}. ',
    'sem22-task6-sarcasm': ' It was {"mask"}. '

}


class BertweetVerbalizerGenerator(VerbalizerGenerator):
    def invalid_label_word(self, word: str):
        return ('@@' in word)

    def post_process(self, word: str):
        return word.lstrip('Ġ')


class ManualTemplateWithoutParse(ManualTemplate):
    """The generated template from TemplateGenerator is a list of dict of parsed template_text. So no further parsing is needed."""

    def on_text_set(self):
        pass

class T5TemplateGenerator(TemplateGenerator):
    def get_part_token_id(self, part_id):
        return self.tokenizer.additional_special_tokens_ids[part_id]
    def convert_template(self, generated_template: List[str], original_template: List[Dict]) -> str:
        r"""
        Given original template used for template generation,convert the generated template into a standard template for downstream prompt model, return a ``str``
        Example:
        generated_template: ['<extra_id_0>', 'it', 'is', '<extra_id_1>', 'one', '</s>']
        original_template: [{'add_prefix_space': '', 'placeholder': 'text_a'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': ' ', 'meta': 'labelword'}, {'add_prefix_space': ' ', 'mask': None}, {'add_prefix_space': '', 'text': '.'}]
        return: "{'placeholder':'text_a'} it is {"mask"} one."
        """
        i = 0
        part_id = 0
        while generated_template[i] != self.tokenizer.additional_special_tokens[part_id] and i < len(generated_template) - 1:
            i += 1
        assert generated_template[i] == self.tokenizer.additional_special_tokens[part_id], print('invalid generated_template {}, missing token {}'.format(generated_template, self.tokenizer.additional_special_tokens[part_id]))
        i += 1

        output = []
        for d in original_template:
            if 'mask' in d:
                j = i + 1
                part_id += 1
                while generated_template[j] != self.tokenizer.additional_special_tokens[part_id] and j < len(generated_template) - 1:
                    j += 1
                output.append(d.get('add_prefix_space', '') + self.tokenizer.convert_tokens_to_string(generated_template[i:j]))
                i = j + 1
            elif 'meta' in d and d['meta'] == 'labelword':
                output.append(d.get('add_prefix_space', '') + '{"mask"}')
            elif 'text' in d:
                output.append(d.get('add_prefix_space', '') + d['text'])
            else:
                prefix = d.get('add_prefix_space', '')
                if 'add_prefix_space' in d:
                    d.pop('add_prefix_space')
                output.append(prefix + json.dumps(d))
        return ''.join(output)
    def _get_templates(self):
        inner_model = self.model
        input_ids = self.input_ids_buffer
        attention_mask = self.attention_mask_buffer

        ori_decoder_input_ids = torch.zeros((input_ids.size(0), self.max_length)).long()
        ori_decoder_input_ids[..., 0] = inner_model.config.decoder_start_token_id


        # decoder_input_ids: decoder inputs for next regressive generation
        # ll: log likelihood
        # output_id: which part of generated contents we are at
        # output: generated content so far
        # last_length (deprecated): how long we have generated for this part
        current_output = [{'decoder_input_ids': ori_decoder_input_ids, 'll': 0, 'output_id': 1, 'output': [], 'last_length': -1}]
        for i in tqdm(range(self.max_length - 2)):
            new_current_output = []
            for item in current_output:
                if item['output_id'] > self.target_number:
                    # Enough contents
                    new_current_output.append(item)
                    continue
                decoder_input_ids = item['decoder_input_ids']

                # Forward
                batch_size = 16
                turn = input_ids.size(0) // batch_size
                if input_ids.size(0) % batch_size != 0:
                    turn += 1
                # aggr_output = []
                for t in range(turn):
                    start = t * batch_size
                    end = min((t + 1) * batch_size, input_ids.size(0))

                    with torch.no_grad():
                        tmp = self.model(input_ids[start:end], attention_mask=attention_mask[start:end], decoder_input_ids=decoder_input_ids.to(input_ids.device)[start:end])[0]#.to(torch.float16)
                    if t == 0:
                        aggr_output = tmp
                    else:
                        aggr_output = torch.cat((aggr_output,tmp), 0)

                # aggr_output = torch.cat(aggr_output, 0)
                # Gather results across all input sentences, and sort generated tokens by log likelihood
                aggr_output = aggr_output.mean(0)
                log_denominator = torch.logsumexp(aggr_output[i], -1).item()
                ids = list(range(inner_model.config.vocab_size))
                ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
                ids = ids[:self.beam_width+3]

                for word_id in ids:
                    output_id = item['output_id']

                    if word_id == self.get_part_token_id(output_id) or word_id == self.tokenizer.eos_token_id:
                        # Finish one part
                        if self.length_limit is not None and item['last_length'] < self.length_limit[output_id - 1]:
                            check = False
                        else:
                            check = True
                        output_id += 1
                        last_length = 0
                    else:
                        last_length = item['last_length'] + 1
                        check = True

                    output_text = item['output'] + [word_id]
                    ll = item['ll'] + aggr_output[i][word_id] - log_denominator
                    new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                    new_decoder_input_ids[:] = decoder_input_ids
                    new_decoder_input_ids[..., i + 1] = word_id

                    if word_id in self.forbidden_word_ids:
                        check = False

                    # Forbid continuous "."
                    if len(output_text) > 1 and output_text[-2] == self.sent_end_id and output_text[-1] == self.sent_end_id:
                        check = False

                    if check:
                        # Add new results to beam search pool
                        new_item = {'decoder_input_ids': new_decoder_input_ids, 'll': ll, 'output_id': output_id, 'output': output_text, 'last_length': last_length}
                        new_current_output.append(new_item)

            if len(new_current_output) == 0:
                break

            new_current_output.sort(key=lambda x: x['ll'], reverse=True)
            new_current_output = new_current_output[:self.beam_width]
            current_output = new_current_output

        return [self.tokenizer.convert_ids_to_tokens(item['output']) for item in current_output]

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        # default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm',
        # default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance',
        default='eval-stance_demo,eval-emotion_demo,eval-irony_demo,eval-offensive_demo,eval-hate_demo,sem21-task7-humor_demo,sem22-task6-sarcasm_demo',
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
        "--seed", default='0', type=str, help="random seed for initialization")
    parser.add_argument(
        "--generate_tmp", default=0, type=int, help="template generation")
    parser.add_argument(
        "--generate_word", default=0, type=int, help="label word generation")
    parser.add_argument(
        "--beam", default=10, type=int, help="label word generation")
    parser.add_argument(
        "--word", default=50, type=int, help="label word generation")
    parser.add_argument(
        "--shot", default='2000', type=str, help="samples for generation")
    parser.add_argument(
        "--demo", default=1, type=int, help="samples for demo")
    parser.add_argument(
        "--name", default='roberta', type=str, help="write name")
    parser.add_argument(
        "--pre_tmp", default=[], nargs='+', help="write name")
    parser.add_argument(
        "--pre_word", default=[], nargs='+', help="write name")
    parser.add_argument(
        "--small", default=0, type=int, help="t5 base")
    parser.add_argument(
        "--max", default=20, type=int, help="t5 base")
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
                                            tokenizer_wrapper_class=MLMTokenizerWrapper, shuffle=True, max_seq_length=128,
                                            batch_size=args.batch_size)
        valid_dataloader = PromptDataLoader(dataset['dev'], template, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
                                            batch_size=args.batch_size)

        model = PromptForClassification(copy.deepcopy(plm).cuda(), template, verbalizer)

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
    return np.mean(score_all)


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
            text_b = ''
            for tmp in range(args.demo):
                text_b += data['text0'] + ' '
            input_example = InputExample(text_a=data['text'], text_b=text_b, label=int(label2idx[data['labels']]))
            dataset[split].append(input_example)

    ##################load models

    if 'roberta' in args.model_name_or_path:
        plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
        # plm = plm.cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        plm = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)#.cuda()
        wrapped_tokenizer = MLMTokenizerWrapper(tokenizer=tokenizer, truncate_method="head", max_seq_length=128)
    verbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx.keys()),
                                  label_words=WORDS[args.task])
    if len(args.pre_tmp) < 1:
        template_text = '{"placeholder":"text_b"} {"placeholder":"text_a"}' + TEMPLATE[args.task]
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
                                      batch_size=args.batch_size, max_seq_length=128)
        for data in dataloader:
            data = data.cuda()
            verbalizer_generator.register_buffer(data)
        label_words_list = verbalizer_generator.generate()
        label_words_list.append(WORDS[args.task])
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

    if args.generate_tmp == 1:
        template_texts_all = []
        for tmp_txt in ['{"mask"} {"mask"} {"mask"} {"placeholder":"text_b"} {"mask"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"mask"} {"mask"} {"placeholder":"text_b"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"mask"} {"mask"} {"placeholder":"text_b"} {"mask"} {"placeholder":"text_a"} ',
                        '{"mask"} {"mask"} {"mask"} {"placeholder":"text_b"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"mask"} {"placeholder":"text_b"} {"mask"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"mask"} {"placeholder":"text_b"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"mask"} {"placeholder":"text_b"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"mask"} {"placeholder":"text_b"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"placeholder":"text_b"} {"mask"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"placeholder":"text_b"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"placeholder":"text_b"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"mask"} {"placeholder":"text_b"} {"placeholder":"text_a"} ', \
                        '{"placeholder":"text_b"} {"mask"} {"mask"} {"mask"} {"placeholder":"text_a"} ', \
                        '{"placeholder":"text_b"} {"mask"} {"mask"} {"placeholder":"text_a"} ',
                        '{"placeholder":"text_b"} {"mask"} {"placeholder":"text_a"} ', \
                        ]:
            tmp_txt1 = tmp_txt + TEMPLATE[args.task].replace('{"mask"}', '{"meta":"labelword"}')
            # for seed in args.seed.split(','):
            #     set_seed(int(seed))
            if args.small == 1:
                template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm(
                    't5', 't5-base')
            else:
                template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper = load_plm(
                    't5', 't5-large')
            template = LMBFFTemplateGenerationTemplate(tokenizer=template_generate_tokenizer, verbalizer=verbalizer,
                                                       text=tmp_txt1)
            # wrapped_example = template.wrap_one_example(dataset['train'][0])
            # print(wrapped_example)

            #################generate template
            print('performing auto_t...')
            template_generate_model = template_generate_model.cuda()
            template_generator = T5TemplateGenerator(template_generate_model, template_generate_tokenizer,
                                                     template_tokenizer_wrapper, verbalizer, beam_width=args.beam,
                                                     target_number=tmp_txt1.count('{"mask"}'), max_length=args.max)


            if args.shot != 'full':
                sampler = FewShotSampler(num_examples_per_label=int(args.shot))
                dataset_gen = sampler(dataset['train'])
            else:
                dataset_gen = dataset['train']
            dataloader = PromptDataLoader(dataset_gen, template, tokenizer=template_generate_tokenizer,
                                          tokenizer_wrapper_class=template_tokenizer_wrapper,
                                          batch_size=args.batch_size,
                                          decoder_max_length=128, max_seq_length=128, shuffle=False,
                                          teacher_forcing=False)
            for data in dataloader:
                data = data.cuda()
                template_generator._register_buffer(data)
            template_generate_model.eval()
            # print('generating...')
            template_texts_format = template_generator._get_templates()
            original_template = template.text
            template_texts = []
            for template_text in template_texts_format:
                try:
                    template_text1 = template_generator.convert_template(template_text, original_template)
                    template_texts.append(template_text1)
                except:
                    print(template_text)

            # template_generator._show_template()
            template_generator.release_memory()
            template_texts_all.extend(template_texts)
            del template_generate_model, template_generate_tokenizer, template_generate_model_config, template_tokenizer_wrapper, template_generator, dataloader,data
            torch.cuda.empty_cache()
        template_texts_all.append('{"placeholder":"text_b"} {"placeholder":"text_a"}' + TEMPLATE[args.task])
        template_texts_all = list(set(template_texts_all))

    else:
        if len(args.pre_tmp) < 1:
            template_texts_all = ['{"placeholder":"text_b"} {"placeholder":"text_a"}' + TEMPLATE[args.task]]
        else:
            template_texts_all = args.pre_tmp

    template_texts = template_texts_all
    print('all the templates:')
    print(template_texts)

    #####################evaluate template
    best_metrics = 0.0
    if len(template_texts) > 1:
        best_template_text = None
        for template_text in tqdm(template_texts):
            score = evaluate_tmp_word(tokenizer, template_text, verbalizer, args, dataset, copy.deepcopy(plm))
            with open(args.name, 'a', encoding='utf-8') as f:
                f.write(template_text + '\n')
                f.write('{:.5f}'.format(score) + '\n')
            if score > best_metrics:
                print('current best score:', score)
                best_metrics = score
                best_template_text = template_text
    else:
        best_template_text = template_texts[0]
    # use the best template
    template_text = best_template_text
    template = ManualTemplate(tokenizer, text=best_template_text)
    print("final best template:")
    print(best_template_text)
    # print("wrapped example:", template.wrap_one_example(dataset["train"][0]))
    with open(args.name, 'a', encoding='utf-8') as f:
        f.write(best_template_text + '\n')
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

