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
    AutoModelForMaskedLM,
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

CONVERT = {
    'stance':{'NONE':0,'FAVOR':1,'AGAINST':2},
    'hate': {'normal':0,'hatespeech':1,'offensive':2},
    'sem-18':{'0':0,'1':1},
    'sem-17':{'neutral':0,'positive':1,'negative':2},
    'imp-hate':{'not_hate':0,'implicit_hate':1,'negative':2},
    'sem19-task5-hate':{'0':0,'1':1},
    'sem19-task6-offen':{'NOT':0,'OFF':1},
    'sem22-task6-sarcasm':{'0':0,'1':1}
}



WORDS = {
    'stance':[["yes"], ["agree","like","favor"], ["dis","don't","not","hate"]],
    'hate':'{"placeholder":"text_a"} I {"mask"} this.',
    'sem-18':'{"placeholder":"text_a"} I {"mask"} this.',
    'sem-17':'{"placeholder":"text_a"} I {"mask"} this.',
    'imp-hate':'{"placeholder":"text_a"} I {"mask"} this.',
    'sem19-task5-hate':'{"placeholder":"text_a"} I {"mask"} this.',
    'sem19-task6-offen':'{"placeholder":"text_a"} I {"mask"} this.',
    'sem22-task6-sarcasm':'{"placeholder":"text_a"} I {"mask"} this.',
}

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
        "--template",
        default='I {"mask"} this.',
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
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=30,#10 for prompt
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
        "--shot",
        type=int,
        default=10,
        help="Save checkpoint every X updates steps.")
    args = parser.parse_args()
    return args

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        batch = batch.cuda()
        logits = model(batch)
        preds = logits.argmax(axis=1)
        label_all.extend(batch['label'].cpu().tolist())
        pred_all.extend(preds.cpu().tolist())

    f1_ma = f1_score(label_all, pred_all,average='macro')
    f1_mi = f1_score(label_all, pred_all, average='micro')
    f1_we = f1_score(label_all, pred_all, average='weighted')
    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (f1_ma, f1_mi, f1_we))
    return f1_ma, f1_mi, f1_we

import json
def read_data(fileName):
    with open(fileName, 'r') as f:
        data = f.readlines()
    data_dic = []
    for line in data:
        data_dic.append(json.loads(line))
    return data_dic

def read_label(data):
    label_name = set()
    for one in data:
        label_name.add(one['labels'])
    label2idx = {}
    label_name = sorted(list(label_name))
    for idx in range(0, len(label_name)):
        label2idx[label_name[idx]] = idx
    return label2idx

def do_train(args):
    # set_seed(args.seed)
    print(args)
    TEMPLATE = {
        'stance': '{"placeholder":"text_a"} '+args.template,
        'hate': '{"placeholder":"text_a"} '+args.template,
        'sem-18': '{"placeholder":"text_a"} '+args.template,
        'sem-17': '{"placeholder":"text_a"} '+args.template,
        'imp-hate': '{"placeholder":"text_a"} '+args.template,
        'sem19-task5-hate': '{"placeholder":"text_a"} '+args.template,
        'sem19-task6-offen': '{"placeholder":"text_a"} '+args.template,
        'sem22-task6-sarcasm': '{"placeholder":"text_a"} '+args.template,
    }
    accelerator = Accelerator()
    ####################dataset
    from openprompt.data_utils import InputExample
    label2idx = CONVERT[args.task]
    dataset = {}
    for split in ['train', 'dev', 'test']:
        dataset[split] = []
        data_all = read_data(args.input_dir+split+'.json')
        for data in data_all:
            input_example = InputExample(text_a=data['text'], label=int(label2idx[data['labels']]))
            dataset[split].append(input_example)
    ##################few shot
    if args.shot:
        from openprompt.data_utils.data_sampler import FewShotSampler
        sampler = FewShotSampler(num_examples_per_label=args.shot)
        dataset['train'] = sampler(dataset['train'])
    #####################models
    from openprompt.plms.mlm import MLMTokenizerWrapper
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    plm = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
    wrapped_t5tokenizer = MLMTokenizerWrapper(max_seq_length=128,tokenizer=tokenizer)

    ######################template
    from openprompt.prompts import ManualTemplate
    template_text = TEMPLATE[args.task]
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
    # wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
    # print(wrapped_example)
    # tokenized_example = wrapped_t5tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
    # print(tokenized_example)
    # print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))
    # print(tokenizer.convert_ids_to_tokens(tokenized_example['decoder_input_ids']))

    ##################data loader
    from openprompt import PromptDataLoader
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
                                        batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                        predict_eos_token=False, truncate_method="tail")
    dev_dataloader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
                                      batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                      predict_eos_token=False, truncate_method="tail")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=MLMTokenizerWrapper, max_seq_length=128,
                                      batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                      predict_eos_token=False, truncate_method="tail")

    ############################sampler
    # from openprompt.data_utils.data_sampler import FewShotSampler
    # sampler = FewShotSampler(num_examples_per_label=16, num_examples_per_label_dev=16, also_sample_dev=True)
    # dataset['train'], dataset['validation'] = sampler(dataset['train'])

    ####################verbalizer
    # from openprompt.prompts import ManualVerbalizer
    # import torch
    # myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx),
    #                                 label_words=WORDS[args.task])

    from openprompt.prompts import SoftVerbalizer
    myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(label2idx), label_words=None)

    #######################train
    from openprompt import PromptForClassification
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    prompt_model = prompt_model.cuda()
    from transformers import AdamW, get_linear_schedule_with_warmup
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr": args.learning_rate},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr": args.learning_rate*10},
    ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    # prompt_model, optimizer1,  optimizer2, train_dataloader, dev_dataloader, test_dataloader = accelerator.prepare(
    #     prompt_model, optimizer1,  optimizer2, train_dataloader, dev_dataloader, test_dataloader
    # )
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler1 = get_scheduler(
        name='linear',
        optimizer=optimizer1,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    lr_scheduler2 = get_scheduler(
        name='linear',
        optimizer=optimizer2,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    global_step = 0
    best_metric = [0, 0, 0]
    tic_train = time.time()
    for epoch in range(args.num_train_epochs):
        tot_loss = 0
        for step, inputs in enumerate(train_dataloader):
            inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            lr_scheduler1.step()
            lr_scheduler2.step()
            global_step += 1
        if (epoch + 1) % args.logging_steps == 0:
            print(
                "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                % (global_step, max_train_steps, epoch,
                   loss, args.logging_steps / (time.time() - tic_train),
                   args.seed,float(0.0001),args.input_dir))
            tic_train = time.time()
        if (epoch + 1) % args.save_steps == 0:
            tic_eval = time.time()
            cur_metric = evaluate(prompt_model, dev_dataloader)
            print("eval done total : %s s" % (time.time() - tic_eval))
            if cur_metric[0] > best_metric[0]:
                prompt_model.plm.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                best_metric = cur_metric
    del plm,prompt_model  # , optimizer, logits, logits_seq, loss, loss_seq, loss_all, accelerator
    torch.cuda.empty_cache()
    plm = AutoModelForMaskedLM.from_pretrained(args.output_dir, config=config)
    prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    prompt_model = prompt_model.cuda()

    cur_metric = evaluate(prompt_model, test_dataloader)
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
    del plm,prompt_model
    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    for task in args.task_name.split(','):
        for model_name in args.model_name_or_path.split(','):
            ave_metric = []
            for seed in args.seed.split(','):
                set_seed(int(seed))
                args_tmp = copy.deepcopy(args)
                args_tmp.task = task
                args_tmp.input_dir = args.input_dir + task + '/'
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
