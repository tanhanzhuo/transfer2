import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='./', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='/work/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='twitter_hash', type=str, required=False, help="dataset name")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
parser.add_argument("--tokenizer_name", default='vinai/bertweet-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=10, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = set()
        lines = f.readlines()
        for line in lines:
            line_clean = line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()
            line_no = line_clean.replace('@USER', '').replace('https', '').replace(' ','')
            if len(line_no) > 10:
                data.add(line_clean)
    return data

def write_data(fileName):
    data = list(read_data(fileName+'.txt'))
    with open(fileName+'_clean.txt', 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one + ' \n')

def tokenization(args):
    # if args.output_dir + args.task_name is not None:
    #     os.makedirs(args.output_dir+ args.task_name, exist_ok=True)
    # Get the datasets:
    # if args.dataset_path is not None:
    if not os.path.isfile(args.dataset_path + args.task_name +'_clean.txt'):
        write_data(args.dataset_path + args.task_name)
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '_clean.txt'
    raw_datasets = datasets.load_dataset('text', data_files=data_files)
    raw_datasets["train"] = raw_datasets["train"].shuffle()
    # Load pretrained tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer,
                                                  normalization=True)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = False

    def tokenize_function(examples):
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
    args_tmp = copy.deepcopy(args)
    tokenized_datasets = tokenization(args_tmp)
    tokenized_datasets.save_to_disk(args_tmp.output_dir + args.task_name)