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
import os
import random
import time
import math
from functools import partial

import numpy as np
import paddle
from paddle.io import DataLoader

import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
# from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer
from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
import copy
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report

from paddlenlp.transformers import RobertaTokenizer
from transformers import AutoTokenizer
from modeling import RobertaForTokenClassification

# os.environ["CUDA_VISIBLE_DEVICES"]="3"
MODEL_CLASSES = {
    "bert": (BertForTokenClassification, BertTokenizer),
    "ernie": (ErnieForTokenClassification, ErnieTokenizer),
}
parser = argparse.ArgumentParser()
IGNORE = -100

# yapf: disable
parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--token_name_or_path", default='bert-base-uncased', type=str, required=False, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))

parser.add_argument("--input_dir", default='./tmp/', type=str, required=False, help="The input directory where the model predictions and checkpoints will be written.")
parser.add_argument("--output_dir", default='./tmp/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=30, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=1000, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=1, help="Save checkpoint every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--device", default="gpu", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
# yapf: enable

def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)


def evaluate(model, loss_fct, data_loader, tokenizer):
    model.eval()
    f = open('/work/test/pretrain_hashtag/keyphrase/twitter_hash_key/test3m_pred.txt', 'w')
    label_all = []
    pred_all = []
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        # loss = loss_fct(logits, labels)
        preds = logits.argmax(axis=2).cpu().numpy()
        labels = labels.cpu().numpy()
        for idx in range(len(input_ids)):
            pred_one = preds[idx][:length[idx]]
            label_one = labels[idx][:length[idx]]
            input_one = input_ids[idx].cpu().numpy()[:length[idx]]
            f.write(tokenizer.decode(input_one) + '\n')
            f.write(tokenizer.decode(input_one[label_one==1]) + '\n')
            f.write(tokenizer.decode(input_one[pred_one==1]) + '\n')
            f.write('\n')
            label_all.extend(label_one)
            pred_all.extend(pred_one)

    f.close()
    names = ['neg', 'pos']
    rep = classification_report(label_all, pred_all,
                            target_names=names, digits=5, output_dict=True)
    # aveRec =  rep['macro avg']['recall']
    # f1PN = (rep['pos']['f1-score'] + rep['neg']['f1-score'])/2
    # acc = rep['accuracy']
    f1 = rep['pos']['f1-score']
    print("Precision:%.5f, Recall:%.5f, F1: %.5f " % (rep['pos']['precision'], rep['pos']['recall'], rep['pos']['f1-score']))
    model.train()
    return rep['pos']['f1-score']
    # return aveRec, f1PN, acc

def tokenize_and_align_labels(example, tokenizer,
                              max_seq_len=128):
    tokenized_example = {'input_ids':[0,2], 'token_type_ids':[0,0],'seq_len':2, 'labels':[0,0]}
    line =   example['sentence']
    input_ids = tokenizer(line)['input_ids']
    if len(input_ids) >= 128:
        # print('error*********************11111111')
        return tokenized_example

    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    # eos_token_id = tokenizer.sep_token_id
    LEN = len(input_ids)
    mask_idx = -1
    for token_id in range(LEN):
        if input_ids[token_id] == mask_token_id:
            mask_idx = token_id
            break
    eos_idx = -1
    for token_id in range(LEN-2, -1, -1):
        if input_ids[token_id] == eos_token_id:
            eos_idx = token_id
            break
    if mask_idx == -1 or eos_idx == -1:
        # print('error*********************22222222')
        return tokenized_example
    
    # hash_sep = tokenizer.decode( input_ids[eos_idx+1:LEN-1] )##############exclude strange phrases
    # if len( hash_sep.replace(' ','') ) <= 5: #################do not split words like axure, lol, cc201
    #     hash_sep = hash_sep.replace(' ','')
    #     hash_token = tokenizer(hash_sep,add_special_tokens=False)['input_ids']
    #     input_ids = input_ids[:mask_idx] + hash_token + input_ids[mask_idx+1:eos_idx+1]
    #     # labels = [-100]*mask_idx + input_ids[eos_idx+1:LEN-1] + [-100]*(eos_idx-mask_idx)
    #     labels = [0]*mask_idx + [1] * (len(hash_token)) + [0]*(eos_idx-mask_idx)
    # else:
    
    input_ids = input_ids[:mask_idx] + input_ids[eos_idx+1:LEN-1] + input_ids[mask_idx+1:eos_idx+1]
    # labels = [-100]*mask_idx + input_ids[eos_idx+1:LEN-1] + [-100]*(eos_idx-mask_idx)
    labels = [0]*mask_idx + [1] * (LEN - eos_idx -2) + [0]*(eos_idx-mask_idx)

    tokenized_example['input_ids'] = input_ids
    tokenized_example['labels'] = labels
    tokenized_example['token_type_ids'] = [0] * len(tokenized_example['input_ids'])
    tokenized_example['seq_len'] = len(tokenized_example['input_ids'])

    return tokenized_example


def read_data(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            yield {'sentence':line}
        f.close()


def read_data_test(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('@USER', '')###################VIP
            line = line.replace('HTTPURL', '')###################VIP
            yield {'sentence':line}
        f.close()

def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    # set_seed(args.seed)
    # Create dataset, tokenizer and dataloader.

    train_ds = load_dataset(read_data, fileName=args.input_dir + 'train3m.txt', lazy=False)
    dev_ds = load_dataset(read_data, fileName=args.input_dir + 'test3m.txt', lazy=False)

    tokenizer = AutoTokenizer.from_pretrained(args.token_name_or_path,normalization=True)
    # tokenizer = ErnieTokenizer.from_pretrained(args.token_name_or_path)

    # Define the model netword and its loss

    model = RobertaForTokenClassification.from_pretrained(
       args.model_name_or_path, num_classes=2)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    trans_func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length)

    dev_ds = dev_ds.map(trans_func)
    train_ds = train_ds.map(trans_func)

    ignore_label = IGNORE

    batchify_fn = lambda samples, fn=Dict({
        'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
        'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
        'seq_len': Stack(dtype='int64'),  # seq_len
        'labels': Pad(axis=0, pad_val=ignore_label, dtype='int64')  # label
    }): fn(samples)

    train_batch_sampler = paddle.io.DistributedBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_data_loader = DataLoader(
        dataset=train_ds,
        # shuffle=True,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_sampler=train_batch_sampler,
        return_list=True)

    dev_data_loader = DataLoader(
        dataset=dev_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        return_list=True)


    num_training_steps = args.max_steps if args.max_steps > 0 else len(
        train_data_loader) * args.num_train_epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_steps)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)
    weight = [1.0, 1.0]
    weight = paddle.to_tensor(np.array(weight), dtype="float32")
    
    loss_fct = paddle.nn.loss.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    global_step = 0
    # last_step = args.num_train_epochs * len(train_data_loader)
    tic_train = time.time()
    best_metric = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_data_loader):
            global_step += 1
            input_ids, token_type_ids, _, labels = batch
            logits = model(input_ids, token_type_ids)
            loss = loss_fct(logits, labels)
            avg_loss = paddle.mean(loss)
            if paddle.distributed.get_rank() == 0 and global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
                # cur_metric = evaluate(model, loss_fct, dev_data_loader, tokenizer)
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        if (epoch+1) % args.save_steps == 0:
            if paddle.distributed.get_rank() == 0:
                
                cur_metric = evaluate(model, loss_fct, dev_data_loader, tokenizer)
                if cur_metric > best_metric:
                    model_to_save = model._layers if isinstance(
                        model, paddle.DataParallel) else model
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    best_metric = cur_metric
    return cur_metric

if __name__ == "__main__":
    args = parser.parse_args()

    r_dir = '/work/test/finetune/continue/'
    for task in ['twitter_hash_key']:      
        for model_name in ['/work/test/pretrain_hashtag/keyphrase/bertweet/']:
            ave_metric = []
            for seed in [1, 10, 100, 1000, 10000]:
                args_tmp = copy.deepcopy(args)
                args_tmp.input_dir = '/work/test/pretrain_hashtag/keyphrase/' + task + '/'
                args_tmp.output_dir = '/work/test/pretrain_hashtag/keyphrase/model/' + task + '/'
                args_tmp.seed = seed
                args_tmp.model_name_or_path = model_name
                args_tmp.token_name_or_path = 'vinai/bertweet-base'
                ave_metric.append(do_train(args_tmp))
            print('final average accuracy: %.5f' % (sum(ave_metric)/5))
            # with open('results_seq_continue_bertweet.txt', 'a') as f_res:
            #     f_res.write(model_name + '\n')
            #     f_res.write('Task: %s, accuracy: %.5f \n' % (task, sum(ave_metric)/5))
            #     f_res.close()
            