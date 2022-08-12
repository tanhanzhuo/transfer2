import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from accelerate import Accelerator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method_hash',default='modelT100N100R_fileT100R_num1_top0_textfirst',type=str)
parser.add_argument('--method_emoji',default='modelT1000000N1000000_file1000000_num10000_top0_textfirst',type=str)
parser.add_argument('--order',default='the',type=str)
parser.add_argument("--tokenizer_name", default='vinai/bertweet-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

def tokenization(args):
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '/train.json'
    data_files["dev"] = args.dataset_path + args.task_name + '/dev.json'
    data_files["test"] = args.dataset_path + args.task_name + '/test.json'
    raw_datasets = datasets.load_dataset('json', data_files=data_files)

    data_files_hash = {}
    data_files_hash["train"] = args.dataset_path + args.task_name + '/train_'  + args.method_hash + '.json'
    data_files_hash["dev"] = args.dataset_path + args.task_name + '/dev_'  + args.method_hash + '.json'
    data_files_hash["test"] = args.dataset_path + args.task_name + '/test_'  + args.method_hash + '.json'
    raw_datasets_hash = datasets.load_dataset('json', data_files=data_files_hash)

    data_files_emoji = {}
    data_files_emoji["train"] = args.dataset_path + args.task_name + '/train_' + args.method_emoji + '.json'
    data_files_emoji["dev"] = args.dataset_path + args.task_name + '/dev_' + args.method_emoji + '.json'
    data_files_emoji["test"] = args.dataset_path + args.task_name + '/test_' + args.method_emoji + '.json'
    raw_datasets_emoji = datasets.load_dataset('json', data_files=data_files_emoji)

    for sp in ['train', 'dev', 'test']:
        train_dataset = raw_datasets[sp]
        text = train_dataset['text']
        train_dataset = train_dataset.remove_columns(['text'])

        text_hash = raw_datasets_hash[sp]['text']
        text_emoji = raw_datasets_emoji[sp]['text']

        text_new = []
        for idx in range(len(text)):
            one = text[idx].strip()
            one_hash = text_hash[idx].strip()
            one_emoji = text_emoji[idx].strip()
            hash = one_hash.replace(one,'')
            emoji = one_emoji.replace(one,'')
            if args.order == 'the':
                text_new.append(one+' '+hash+' '+emoji+' \n ')
            elif args.order == 'teh':
                text_new.append(one + ' ' + emoji + ' ' + hash + ' \n ')
            elif args.order == 'hte':
                text_new.append(hash + ' ' + one + ' ' + emoji + ' \n ')
            elif args.order == 'het':
                text_new.append(hash + ' ' + emoji + ' ' + one + ' \n ')
            elif args.order == 'eth':
                text_new.append(emoji + ' ' + one + ' ' + hash + ' \n ')
            elif args.order == 'eht':
                text_new.append(emoji + ' ' + hash + ' ' + one + ' \n ')
        train_dataset = train_dataset.add_column("text", text_new)

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
    for task in args.task_name.split(','):
            args_tmp = copy.deepcopy(args)
            args_tmp.task_name = task
            tokenized_datasets = tokenization(args_tmp)
            save_hash = args_tmp.method_hash.split('/')[-1]
            save_hash = save_hash.split('top')[0]
            save_emoji = args_tmp.method_emoji.split('/')[-1]
            save_emoji = save_emoji.split('top')[0]

            tokenized_datasets.save_to_disk(args_tmp.dataset_path + args_tmp.task_name + '/emojihash_' \
                                            + save_emoji + save_hash + '_order_'+args_tmp.order)