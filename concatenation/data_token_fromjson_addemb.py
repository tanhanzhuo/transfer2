import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from accelerate import Accelerator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='/work/transfer2/finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='/work/transfer2/finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method',default='modelT100N100S_fileT100S_num1_cluster_top0_hashlast',type=str)
parser.add_argument("--tokenizer_name", default='vinai/bertweet-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")
parser.add_argument('--hash_file',default='../contrastive/feature_modelT100N100S_fileT100S_num1_cluster',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory

def tokenization(args, hash_dic):
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '/train_'  + args.method + '.json'
    data_files["dev"] = args.dataset_path + args.task_name + '/dev_'  + args.method + '.json'
    data_files["test"] = args.dataset_path + args.task_name + '/test_'  + args.method + '.json'
    raw_datasets = datasets.load_dataset('json', data_files=data_files)

    for sp in ['train', 'dev', 'test']:
        train_dataset = raw_datasets[sp]
        text = train_dataset['text']
        train_dataset = train_dataset.remove_columns(['text'])
        embs = []
        text_new = []
        for one in text:
            one_sp = one.strip().split(' ')
            hash = one_sp[-1]
            text_new.append(' '.join(one_sp[:-1]))
            embs.append(hash_dic[hash])
        train_dataset = train_dataset.add_column("text", text_new)
        train_dataset = train_dataset.add_column("embs", embs)
        raw_datasets[sp] = train_dataset.shuffle()

    # Load pretrained tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,normalization=True)

    # First we tokenize all the texts.

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = False

    def tokenize_function(examples):
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=False,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    # tokenized_datasets.save_to_disk(args.output_dir)
    return tokenized_datasets

if __name__ == "__main__":
    args = parser.parse_args()

    hash_embs = []
    hash_tags = []
    for idx in range(args.split):
        tmp = np.load(args.hash_file + '_' + str(idx) + '.npz', allow_pickle=True)
        hash_embs.extend(tmp['center_embs'])
        hash_tags.extend(tmp['center_hash'])
        tmp.close()
    hash_dic = dict(zip(hash_tags, hash_embs))
    for task in args.task_name.split(','):
        for method in args.method.split(','):
            args_tmp = copy.deepcopy(args)
            args_tmp.task_name = task
            args_tmp.method = method
            tokenized_datasets = tokenization(args_tmp, hash_dic)
            tokenized_datasets.save_to_disk(args_tmp.dataset_path + args_tmp.task_name + '/' + args_tmp.hash_file)