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
    get_linear_schedule_with_warmup,
    RobertaForMaskedLM
)
from transformers.modeling_outputs import MaskedLMOutput
from accelerate import Accelerator
from tqdm import trange,tqdm
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

CONVERT = {
    'eval-emotion':{'0':0,'1':1,'2':2,'3':3},
    'eval-hate':{'0':0,'1':1},
    'eval-irony':{'0':0,'1':1},
    'eval-offensive':{'0':0,'1':1},
    'eval-stance':{'0':0,'1':1,'2':2},
    'stance': {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}


WORDS = {
    # 'stance':[["yes"], ["agree","like","favor"], ["dis","don't","not","hate"]],
    'eval-emotion': [["angerous"], ["joyful"], ["optimistic"],["sad"]],
    'eval-hate': [["neutral"], ["hateful"]],
    'eval-irony': [["neutral"], ["ironic"]],
    'eval-offensive': [["neutral"], ["offensive"]],
    'eval-stance': [["neutral"], ["against"],["favor"]],
    'stance': [["neutral"], ["favor"],["against"]],
    'sem22-task6-sarcasm': [["neutral"], ["sarcastic"]],
    'sem21-task7-humor': [["neutral"], ["humorous"]]
}

TEMPLATE = {
    'eval-emotion':' It was {"mask"}. ',
    'eval-hate':' It was {"mask"}. ',
    'eval-irony':' It was {"mask"}. ',
    'eval-offensive':' It was {"mask"}. ',
    'eval-stance':' It was {"mask"}. ',
    'stance':' It was {"mask"}. ',
    'sem22-task6-sarcasm':' It was {"mask"}. ',
    'sem21-task7-humor':' It was {"mask"}. '
}


# class RobertaForMulti(RobertaPreTrainedModel):
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config, add_pooling_layer=False)
#         # self.resize_position_embeddings(max_position)
#         self.lm_head = RobertaLMHead(config)
#
#         self.post_init()
#     def _init_weights(self, module: nn.Module):
#         """Initialize the weights."""
#         if isinstance(module, nn.Linear):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#     ##resize position embedding
#     def resize_position_embeddings(self, new_num_position_embeddings: int):
#         num_old = self.roberta.config.max_position_embeddings
#         if num_old == new_num_position_embeddings:
#             return
#         self.roberta.config.max_position_embeddings = new_num_position_embeddings
#         # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
#         new_position = nn.Embedding(self.roberta.config.max_position_embeddings, self.roberta.config.hidden_size)
#         new_position.to(self.roberta.embeddings.position_embeddings.weight.device,
#                         dtype=self.roberta.embeddings.position_embeddings.weight.dtype)
#         # self._init_weights(new_position)
#         new_position.weight.data[:num_old, :] = self.roberta.embeddings.position_embeddings.weight.data[:num_old, :]
#         self.roberta.embeddings.position_embeddings = new_position
#
#         self.roberta.embeddings.register_buffer("position_ids", torch.arange(self.roberta.config.max_position_embeddings).expand((1, -1)))
#         self.roberta.embeddings.register_buffer(
#             "token_type_ids", torch.zeros([1,self.roberta.config.max_position_embeddings], dtype=torch.long), persistent=False
#         )
#         # with torch.no_grad():
#         #     # self.roberta.embeddings.position_embeddings.weight[:num_old,:] = nn.Parameter(
#         #     #     old_position_embeddings_weight)
#
#     def resize_type_embeddings(self, new_type_embeddings: int):
#         num_old = self.roberta.config.type_vocab_size
#         if num_old == new_type_embeddings:
#             return
#         self.roberta.config.type_vocab_size = new_type_embeddings
#         # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
#         new_type = nn.Embedding(self.roberta.config.type_vocab_size, self.roberta.config.hidden_size)
#         new_type.to(self.roberta.embeddings.token_type_embeddings.weight.device,
#                         dtype=self.roberta.embeddings.token_type_embeddings.weight.dtype)
#         # self._init_weights(new_position)
#         new_type.weight.data[:num_old, :] = self.roberta.embeddings.token_type_embeddings.weight.data[:num_old, :]
#         self.roberta.embeddings.token_type_embeddings = new_type
#
#     def forward(self,
#                 input_ids,
#                 token_type_ids=None,
#                 position_ids=None,
#                 attention_mask=None,
#                 output_hidden_states=None):
#
#
#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             #return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#         prediction_scores = self.lm_head(sequence_output)
#         masked_lm_loss = None
#         return MaskedLMOutput(
#             loss=masked_lm_loss,
#             logits=prediction_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class RobertaForMulti(RobertaForMaskedLM):
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        num_old = self.roberta.config.max_position_embeddings
        if num_old == new_num_position_embeddings:
            return
        self.roberta.config.max_position_embeddings = new_num_position_embeddings
        # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
        new_position = nn.Embedding(self.roberta.config.max_position_embeddings, self.roberta.config.hidden_size)
        new_position.to(self.roberta.embeddings.position_embeddings.weight.device,
                        dtype=self.roberta.embeddings.position_embeddings.weight.dtype)
        # self._init_weights(new_position)
        new_position.weight.data[:num_old, :] = self.roberta.embeddings.position_embeddings.weight.data[:num_old, :]
        self.roberta.embeddings.position_embeddings = new_position

        self.roberta.embeddings.register_buffer("position_ids", torch.arange(self.roberta.config.max_position_embeddings).expand((1, -1)))
        self.roberta.embeddings.register_buffer(
            "token_type_ids", torch.zeros([1,self.roberta.config.max_position_embeddings], dtype=torch.long), persistent=False
        )
        # with torch.no_grad():
        #     # self.roberta.embeddings.position_embeddings.weight[:num_old,:] = nn.Parameter(
        #     #     old_position_embeddings_weight)

    def resize_type_embeddings(self, new_type_embeddings: int):
        num_old = self.roberta.config.type_vocab_size
        if num_old == new_type_embeddings:
            return
        self.roberta.config.type_vocab_size = new_type_embeddings
        # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
        new_type = nn.Embedding(self.roberta.config.type_vocab_size, self.roberta.config.hidden_size)
        new_type.to(self.roberta.embeddings.token_type_embeddings.weight.device,
                        dtype=self.roberta.embeddings.token_type_embeddings.weight.dtype)
        # self._init_weights(new_position)
        new_type.weight.data[:num_old, :] = self.roberta.embeddings.token_type_embeddings.weight.data[:num_old, :]
        self.roberta.embeddings.token_type_embeddings = new_type


from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None


    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for idx_fea in range(len(features)):
            feature = features[idx_fea]
            flat_features.append({k: feature[k][0] if k in special_keys else feature[k] for k in feature})
            flat_features[idx_fea]['input_ids'] = sum(feature['input_ids'][1:],flat_features[idx_fea]['input_ids'])
            flat_features[idx_fea]['attention_mask'] = flat_features[idx_fea]['attention_mask'] + \
                                                       [1]*(len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['attention_mask']))
            flat_features[idx_fea]['token_type_ids'] = flat_features[idx_fea]['token_type_ids'] + \
                                                       [self.tokenizer.pad_token_type_id] * (len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['token_type_ids']))

            # for i in range(num_sent):
                # flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
        #          for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]

        return batch

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        # default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm',
        #default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance',
        default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor',
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
        default='_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp',
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
        default=514,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--token_type",
        default=1,
        type=int)
    parser.add_argument(
        "--learning_rate",
        default='1e-5',#'1e-3,1e-4,1e-5,1e-6',
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
        "--shot", default='full',#'10,20,40,80,160,320,640,1280,full',
        type=str, help="random seed for initialization")
    parser.add_argument(
        "--stop", default=5, type=int, help="early stop")
    parser.add_argument(
        "--weight", default=0, type=int, help="weighted loss")
    parser.add_argument(
        "--write_result", default='', type=str, help="weighted loss")
    parser.add_argument(
        "--demo", default=1, type=int, help="with demo")
    parser.add_argument(
        "--soft", default=1, type=int, help="soft verberlizer")
    args = parser.parse_args()
    return args

def read_data(fileName):
    data_dic = []
    with open(fileName, 'r') as f:
        for line in f:
            data_dic.append(json.loads(line))
    return data_dic

@torch.no_grad()
def evaluate(model, data_loader, task='eval-emoji',write_result=''):
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
            f.write(task+'\n')
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

    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (tweeteval_result, tweeteval_result, tweeteval_result))
    return tweeteval_result,tweeteval_result,tweeteval_result


def do_train(args):
    # print(args)
    # label2idx = CONVERT[args.task]
    # dataset = {}
    # SPLITS = ['train', 'dev', 'test']
    # for split in SPLITS:
    #     dataset[split] = []
    #     data_all = read_data(args.input_dir+split+args.method+'.json')
    #     random.shuffle(data_all)
    #     for data in data_all:
    #         if args.demo:
    #             text_demo = data['text']
    #             for idx in range(len(data.keys())-1):
    #                 text_demo = data['text'+str(idx)] + TEMPLATE[args.task].replace('{"mask"}', WORDS[args.task][str(idx)]) + text_demo
    #             input_example = InputExample(text_a=text_demo, label=int(label2idx[data['labels']]))
    #         else:
    #             input_example = InputExample(text_a=data['text'], label=int(label2idx[data['labels']]))
    #         dataset[split].append(input_example)
    # ##################few shot
    # if args.shot:
    #     sampler = FewShotSampler(num_examples_per_label=args.shot)
    #     dataset['train'] = sampler(dataset['train'])
    #
    # config = AutoConfig.from_pretrained(args.model_name_or_path)
    # plm = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
    # tokenizer = AutoTokenizer.from_pretrained(args.token_name_or_path, normalization=True)
    # wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=128,tokenizer=tokenizer, truncate_method="head")
    #
    # template_text = '{"placeholder":"text_a"}' + TEMPLATE[args.task]
    # mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    #
    # train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    #                                     tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
    #                                     batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    #                                     predict_eos_token=False, truncate_method="head")
    # dev_dataloader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, tokenizer=tokenizer,
    #                                   tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
    #                                   batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    #                                   predict_eos_token=False, truncate_method="head")
    # test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    #                                    tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
    #                                    batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    #                                    predict_eos_token=False, truncate_method="head")
    #
    # myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx),
    #                                 label_words=WORDS[args.task])
    # prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    # prompt_model = prompt_model.cuda()
    #
    # loss_func = torch.nn.CrossEntropyLoss()
    # no_decay = ['bias', 'LayerNorm.weight']
    # # it's always good practice to set no decay to biase and LayerNorm parameters
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}
    # ]
    #
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr)


    print(args)
    label2idx = CONVERT[args.task]
    dataset = {}
    SPLITS = ['train', 'dev', 'test']
    for split in SPLITS:
        dataset[split] = []
        data_all = read_data(args.input_dir+split+args.method+'.json')
        random.shuffle(data_all)
        for data in data_all:
            if args.demo > 0:
                text_demo = data['text']
                for idx in range(args.demo):
                    text_demo = data['text'+str(idx)] + text_demo
                input_example = InputExample(text_a=text_demo, label=int(label2idx[data['labels']]))
            else:
                input_example = InputExample(text_a=data['text'], label=int(label2idx[data['labels']]))
            dataset[split].append(input_example)
    ##################few shot
    if args.shot != 'full':
        sampler = FewShotSampler(num_examples_per_label=int(args.shot))
        dataset['train'] = sampler(dataset['train'])

    learning_rate = args.learning_rate.split(',')
    best_metric = [0, 0, 0]

    for lr in learning_rate:
        best_metric_lr = [0, 0, 0]
        num_classes = len(label2idx.keys())
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True, model_max_length=args.max_seq_length-2)
        tokenizer.model_max_length = args.max_seq_length-2
        tokenizer._pad_token_type_id = args.token_type - 1
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # plm = RobertaForMaskedLM.from_pretrained(
        #     args.model_name_or_path, config=config).cuda()
        plm = RobertaForMulti.from_pretrained(
            args.model_name_or_path, config=config).cuda()
        plm.resize_position_embeddings(args.max_seq_length)
        plm.resize_type_embeddings(args.token_type)

        # from openprompt.plms import load_plm
        # plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")
        wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=args.max_seq_length-2, tokenizer=tokenizer, truncate_method="head")

        template_text = '{"placeholder":"text_a"}' + TEMPLATE[args.task]
        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

        train_data_loader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                            tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=args.max_seq_length-2,
                                            batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                            predict_eos_token=False, truncate_method="head")
        dev_data_loader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, tokenizer=tokenizer,
                                          tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=args.max_seq_length-2,
                                          batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                          predict_eos_token=False, truncate_method="head")
        test_data_loader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                           tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=args.max_seq_length-2,
                                           batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                           predict_eos_token=False, truncate_method="head")
        if args.soft == 1:
            myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(label2idx.keys()))
        else:
            myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx.keys()),
                                        label_words=WORDS[args.task])
        model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        model = model.cuda()

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
        if args.weight == 1:# or 'sarcasm' in args.task:
            num_dic = {}
            for val in label2idx.values():
                num_dic[val] = 0.0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                num_dic[label_tmp] += 1.0
            num_max = max(num_dic.values())
            class_weights = [num_max / i for i in num_dic.values()]
            class_weights = torch.FloatTensor(class_weights).cuda()
            loss_fct = nn.CrossEntropyLoss(weight=class_weights).cuda()

        print('start Training!!!')
        global_step = 0
        tic_train = time.time()

        stop_sign = 0
        for epoch in trange(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                # input_ids, segment_ids, labels = batch
                # logits = model(input_ids.cuda(), segment_ids.cuda())
                # loss = loss_fct(logits, labels.cuda().view(-1))
                logits = model(batch.cuda())
                loss = loss_fct(logits, batch['label'].cuda().view(-1))
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
            if (epoch + 1) % args.save_steps == 0 and (epoch + 1) > 3:
                tic_eval = time.time()
                cur_metric = evaluate(model, dev_data_loader, args.task)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric_lr[0]:
                    best_metric_lr = cur_metric
                    stop_sign = 0
                    if best_metric_lr[0] > best_metric[0]:
                        model_best = copy.deepcopy(model).cpu()
                        best_metric = best_metric_lr

                else:
                    stop_sign += 1
            if stop_sign >= args.stop:
                break
        del model
        torch.cuda.empty_cache()

    model = model_best.cuda()
    cur_metric = evaluate(model, test_data_loader,args.task,args.write_result)
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
    del model
    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    for shot in args.shot.split(','):
        for task in args.task.split(','):
            for model_name in args.model_name_or_path.split(','):
                ave_metric = []
                for seed in args.seed.split(','):
                    set_seed(int(seed))
                    args_tmp = copy.deepcopy(args)
                    args_tmp.task = task.split('_')[0]
                    args_tmp.input_dir = args.input_dir + task + '/'
                    args_tmp.seed = int(seed)
                    args_tmp.shot = shot
                    args_tmp.model_name_or_path = model_name
                    ave_metric.append(do_train(args_tmp))
                ave_metric = np.array(ave_metric)
                num_seed = len(args.seed.split(','))
                print("*************************************************************************************")
                print('Task: %s, model: %s, shot: %s' % (task, model_name, shot))
                print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:, 0]) / num_seed,
                                                                    sum(ave_metric[:, 1]) / num_seed,
                                                                    sum(ave_metric[:, 2]) / num_seed))
                with open(args.results_name, 'a') as f_res:

                    f_res.write('Task: %s, model: %s, shot: %s\n' % (task, model_name, shot) )
                    f_res.write('aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (sum(ave_metric[:, 0]) / num_seed,
                                                                          sum(ave_metric[:, 1]) / num_seed,
                                                                          sum(ave_metric[:, 2]) / num_seed))
                    for tmp in range(num_seed):
                        f_res.write('%.5f, %.5f, %.5f \n' % (ave_metric[tmp, 0],ave_metric[tmp, 1],ave_metric[tmp, 2]))

                    f_res.close()
