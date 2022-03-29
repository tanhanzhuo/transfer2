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
from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer
from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
import copy
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,classification_report

# os.environ["CUDA_VISIBLE_DEVICES"]="3"
MODEL_CLASSES = {
    "bert": (BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaForTokenClassification, RobertaTokenizer),
    "ernie": (ErnieForTokenClassification, ErnieTokenizer),
}
parser = argparse.ArgumentParser()
IGNORE = -100

# yapf: disable
parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, required=False, help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(list(BertTokenizer.pretrained_init_configuration.keys())))
parser.add_argument("--input_dir", default='./tmp/', type=str, required=False, help="The input directory where the model predictions and checkpoints will be written.")
parser.add_argument("--output_dir", default='./tmp/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=30, type=int, help="Total number of training epochs to perform.", )
parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=3, help="Save checkpoint every X updates steps.")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--device", default="gpu:6", type=str, choices=["cpu", "gpu", "xpu"] ,help="The device to select to train the model, is must be cpu/gpu/xpu.")
# yapf: enable

def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(seed)

@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader, label_num):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            length, preds, labels)
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))
    model.train()
    with open('results_tmp.txt', 'a') as f_res:
        f_res.write('precision: %.5f, recall: %.5f, f1: %.5f \n' % (precision, recall, f1_score) )
        f_res.close()
    return f1_score

@paddle.no_grad()
def evaluate_ner(model, loss_fct, metric, data_loader, label_num):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)

        true_predictions = [
            [p for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(preds.cpu().numpy(), labels.cpu().numpy())
        ]

        true_labels = [
            [l for l in gold_label if l != -100]
            for gold_label in labels.cpu().numpy()
        ]

        true_length = [len(arr) for arr in true_labels]
        
        true_len_max = max(true_length)
        true_predictions_pad = np.array([true_pre + [0]*(true_len_max-true_len) for (true_pre, true_len) in zip(true_predictions,true_length)])
        true_labels_pad = np.array([true_lab + [0]*(true_len_max-true_len) for (true_lab, true_len) in zip(true_labels,true_length)])

        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            paddle.to_tensor(true_length), paddle.to_tensor(true_predictions_pad), paddle.to_tensor(true_labels_pad))
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (avg_loss, precision, recall, f1_score))
    model.train()
    # with open('results_tmp.txt', 'a') as f_res:
    #     f_res.write('precision: %.5f, recall: %.5f, f1: %.5f \n' % (precision, recall, f1_score) )
    #     f_res.close()
    return f1_score


def evaluate_my(model, loss_fct, data_loader):
    model.eval()
    acc, total = 0, 0
    for batch in data_loader:
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = loss_fct(logits, labels)
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2).cpu().numpy()
        labels = labels.cpu().numpy()
        preds[labels == IGNORE] = IGNORE
        acc_num_useless = np.sum(labels == IGNORE)
        acc_num_total = np.sum(labels == preds)
        acc += acc_num_total - acc_num_useless
        total +=  np.sum(labels != IGNORE)
    print('eval loss %.5f, acc: %.5f' % (avg_loss, acc*1.0/total))
    model.train()

    return acc*1.0/total

def tokenize_and_align_labels(example, tokenizer,
                              max_seq_len=128):
    tokenized_example = {'input_ids':[], 'token_type_ids':[],'seq_len':[], 'labels':[]}
    id_all, tag_all = [], []
    for word, tag in zip(example['sentence'], example['label']):
        id_tmp = tokenizer(word)['input_ids']
        if len(id_tmp) <=2:
            continue
        id_tmp = id_tmp[1:-1]
        tag_tmp = [tag] + [IGNORE] * (len(id_tmp) - 1)
        # tag_tmp = [tag] + [tag] * (len(id_tmp) - 1)
        id_all.extend(id_tmp)
        tag_all.extend(tag_tmp)
    id_tmp = tokenizer(word)['input_ids']
    start_id = id_tmp[0]
    end_id = id_tmp[-1]

    # if len(id_all)!=len(tag_all):
    #     print(example)
    #     print(tokenized_example)
    assert len(id_all)==len(tag_all)    
    tokenized_example['input_ids'] = [start_id] + id_all + [end_id]
    tokenized_example['labels'] = [IGNORE] + tag_all + [IGNORE]
    tokenized_example['token_type_ids'] = [0] * len(tokenized_example['input_ids'])
    tokenized_example['seq_len'] = len(tokenized_example['input_ids'])

    return tokenized_example

# def tokenize_and_align_labels_ner(example, tokenizer, label2idx,
#                               max_seq_len=512):
#     tokenized_example = {'input_ids':[], 'token_type_ids':[],'seq_len':[], 'labels':[]}
#     id_all, tag_all = [], []
#     for word, tag in zip(example['sentence'], example['label']):
#         id_tmp = tokenizer(word)['input_ids']
#         if len(id_tmp) <=2:
#             continue
#         id_tmp = id_tmp[1:-1]
#         tag_tmp = [tag] + [label2idx['O']] * (len(id_tmp) - 1)
#         # tag_tmp = [tag] + [tag] * (len(id_tmp) - 1)
#         id_all.extend(id_tmp)
#         tag_all.extend(tag_tmp)
#     id_tmp = tokenizer(word)['input_ids']
#     start_id = id_tmp[0]
#     end_id = id_tmp[-1]

#     # if len(id_all)!=len(tag_all):
#     #     print(example)
#     #     print(tokenized_example)
#     assert len(id_all)==len(tag_all)    
#     tokenized_example['input_ids'] = [start_id] + id_all + [end_id]
#     tokenized_example['labels'] = [label2idx['O']] + tag_all + [label2idx['O']]
#     tokenized_example['token_type_ids'] = [0] * len(tokenized_example['input_ids'])
#     tokenized_example['seq_len'] = len(tokenized_example['input_ids'])

#     return tokenized_example

def read_label(fileName):
    with open(fileName, 'r') as f:
        lines = f.readlines()
        label_name = set()

        for line in lines:
            if line == '\n' or line == '\t\n':
                continue
            label_name.add(line.split()[1])
        f.close()
    label2idx = {}
    label_name = list(label_name)
    for idx in range(0, len(label_name)):
        label2idx[label_name[idx]] = idx
    return label2idx

def read_data(fileName, label2idx):
    with open(fileName, 'r') as f:
        lines = f.readlines()

        sentence = []
        sentence_label = []
        for line in lines:
            if line == '\n' or line == '\t\n':
                yield {'sentence':sentence, 'label':sentence_label}
                sentence = []
                sentence_label = []
            else:
                sentence.append(line.split()[0])
                sentence_label.append(label2idx[line.split()[1]])
        f.close()


def do_train(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
    set_seed(args.seed)
    # Create dataset, tokenizer and dataloader.
    label2idx = read_label(args.input_dir + 'train')
    label_list_final = label2idx.keys()
    if 'ner' in args.input_dir:
        label2idx_order = {}
        label_name = label2idx.keys()
        label_name_set = set()
        for label in label_name:
            if len(label) == 1:
                label_name_for_o = label
            else:
                label_name_set.add(label[2:])

        label_name_list = list(label_name_set)
        label_name_list.append(label_name_for_o)
        label_list_final = []
        for idx in range(0, len(label_name_list)-1):
            label2idx_order['B-' + label_name_list[idx]] = idx * 2
            label2idx_order['I-' + label_name_list[idx]] = idx * 2 +1
            label_list_final.append('B-' + label_name_list[idx])
            label_list_final.append('I-' + label_name_list[idx])
        label2idx_order[label_name_for_o] = (idx+1) * 2
        label_list_final.append(label_name_for_o)
        label2idx = label2idx_order
        
    train_ds = load_dataset(read_data, fileName=args.input_dir + 'train', label2idx=label2idx, lazy=False)
    dev_ds = load_dataset(read_data, fileName=args.input_dir + 'dev', label2idx=label2idx, lazy=False)
    model_class, tokenizer_class = MODEL_CLASSES['ernie']
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    #tokenizer = ErnieCtmTokenizer.from_pretrained("ernie-ctm")

    # Define the model netword and its loss
    model = model_class.from_pretrained(
        args.model_name_or_path, num_classes=len(label2idx.keys()))
    #model = ErnieCtmForTokenClassification.from_pretrained(
    #    args.model_name_or_path, num_classes=label_num)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    trans_func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length)

    # trans_func_ner_test = partial(
    #     tokenize_and_align_labels_ner,
    #     tokenizer=tokenizer,
    #     label2idx=label2idx,
    #     max_seq_len=args.max_seq_length)
    
    # trans_func_ner_test = partial(
    #     tokenize_and_align_labels,
    #     tokenizer=tokenizer,
    #     max_seq_len=args.max_seq_length)


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
    weight = [1.0] * (len(label_list_final) - 1)
    weight.append(0.15)
    weight = paddle.to_tensor(np.array(weight), dtype="float32")
    
    if 'ner' in args.input_dir:
        loss_fct = paddle.nn.loss.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
    else:
        loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=ignore_label)

    metric = ChunkEvaluator(label_list=label_list_final)

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
            if global_step % args.logging_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_step, epoch, step, avg_loss,
                       args.logging_steps / (time.time() - tic_train)))
                tic_train = time.time()
            avg_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
        if (epoch+1) % args.save_steps == 0:
            if paddle.distributed.get_rank() == 0:
                if 'ner' in args.input_dir:
                    cur_metric = evaluate_ner(model, loss_fct, metric, dev_data_loader, len(label2idx.keys()))
                else:
                    cur_metric = evaluate_my(model, loss_fct, dev_data_loader)
                if cur_metric > best_metric:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    best_metric = cur_metric
    model = model_class.from_pretrained(
        args.output_dir, num_classes=len(label2idx.keys()))
    test_ds = load_dataset(read_data, fileName=args.input_dir + 'test', label2idx=label2idx, lazy=False)
    if 'ner' in args.input_dir:
        test_ds = test_ds.map(trans_func)
    else:
        test_ds = test_ds.map(trans_func)

    test_data_loader = DataLoader(
        dataset=test_ds,
        collate_fn=batchify_fn,
        num_workers=0,
        batch_size=args.batch_size,
        return_list=True)
    if 'ner' in args.input_dir:
        cur_metric = evaluate_ner(model, loss_fct, metric, test_data_loader, len(label2idx.keys()))
    else:
        cur_metric = evaluate_my(model, loss_fct, test_data_loader)
    print('best dev: %.5f, test: %.5f' % (best_metric, cur_metric) )
    return cur_metric

if __name__ == "__main__":
    args = parser.parse_args()

    # r_dir = '/work/test/finetune/newtrain/hf/'
    r_dir = '/work/test/hf/collator/pd/fp32/tmp/paddle_'
    for task in ['ner-16', 'ner-17']: #['pos-ritter', 'pos-ark', 'pos-tb', 'ner-16', 'ner-17']:      
        for model_name in [r_dir+'100000/', r_dir+'200000/', r_dir+'300000/', r_dir+'400000/', r_dir+'500000/',\
             r_dir+'600000/', r_dir+'700000/', r_dir+'800000/']:#, r_dir+'900000/', r_dir+'1000000/']:
            ave_metric = []
            for seed in [1, 10, 100, 1000, 10000]:
                args_tmp = copy.deepcopy(args)
                args_tmp.input_dir = '/work/test/finetune/data/' + task + '/'
                args_tmp.output_dir = '/work/test/finetune/model_newtrain/pd/' + task + '/'
                args_tmp.seed = seed
                args_tmp.model_name_or_path = model_name
                ave_metric.append(do_train(args_tmp))
            print('final average accuracy: %.5f' % (sum(ave_metric)/5))
            with open('results_seq_newtrain.txt', 'a') as f_res:
                f_res.write(model_name + '\n')
                f_res.write('Task: %s, accuracy: %.5f \n' % (task, sum(ave_metric)/5))
                f_res.close()
            