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
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

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
# from att import ScaledDotProductAttention
# class RobertaClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#         self.att = ScaledDotProductAttention(d_model=config.hidden_size, d_k=config.hidden_size, d_v=config.hidden_size,
#                                              h=8)
#     def forward(self, features, **kwargs):
#         # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.att(features[:,0:1,:], features, features).squeeze(1)
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, mask, **kwargs):
        x = (features * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1) # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForMulti(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.resize_position_embeddings(max_position)
        self.classifier = RobertaClassificationHead(config)

        self.post_init()
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    ##resize position embedding
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

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                text_mask=None):
        return_dict = False

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # sequence_cls = sequence_cls.sum(dim=1)
        logits = self.classifier(sequence_output,1-text_mask)
        return logits

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

            flat_features[idx_fea]['special_tokens_mask'] = [0] * len(flat_features[idx_fea]['token_type_ids']) + \
                                             [1] * (len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['token_type_ids']))

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
        default='hash_modelT100N100R_fileT100N100R_num10_top_9',
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
        default=2,
        type=int)
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
        default=4,
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
        # input_ids, segment_ids, labels = batch
        # logits = model(input_ids.cuda(), segment_ids.cuda())
        logits = model(input_ids=batch['input_ids'].cuda(),
                       token_type_ids=batch['token_type_ids'].cuda(),
                       attention_mask=batch['attention_mask'].cuda(),
                       text_mask = batch['special_tokens_mask'].cuda())
        labels = batch['labels']
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
    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10)
    # # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # model = RobertaForMulti.from_pretrained(
    #     args.model_name_or_path, config=config).cuda()
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
        if args.shot != 'full':
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
        tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
        tokenizer._pad_token_type_id = args.token_type - 1
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMulti.from_pretrained(
            args.model_name_or_path, config=config).cuda()
        model.resize_position_embeddings(args.max_seq_length)
        model.resize_type_embeddings(args.token_type)
        batchify_fn = OurDataCollatorWithPadding(tokenizer=tokenizer)
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
                # input_ids, segment_ids, labels = batch
                # logits = model(input_ids.cuda(), segment_ids.cuda())
                # loss = loss_fct(logits, labels.cuda().view(-1))
                logits = model(input_ids=batch['input_ids'].cuda(),
                               token_type_ids = batch['token_type_ids'].cuda(),
                               attention_mask=batch['attention_mask'].cuda(),
                               text_mask = batch['special_tokens_mask'].cuda())
                loss = loss_fct(logits, batch['labels'].cuda().view(-1))
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
