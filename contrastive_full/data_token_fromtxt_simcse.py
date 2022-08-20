import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
from tqdm import tqdm
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='./', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='/work/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='twitter_hash', type=str, required=False, help="dataset name")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
parser.add_argument("--tokenizer_name", default='princeton-nlp/sup-simcse-roberta-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=10, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = set()
        lines = f.readlines()
        print('read lines')
        for line in tqdm(lines):
            line_clean = line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()
            line_no = line_clean.replace('@USER', '').replace('https', '').replace(' ','')
            if len(line_no) > 10:
                data.add(line_clean)
    return data

def write_data(fileName):
    data = list(read_data(fileName+'.txt'))
    print('write lines')
    with open(fileName+'_clean.txt', 'w', encoding='utf-8') as f:
        for one in tqdm(data):
            f.write(one + ' \n')

import re
import string
HASH = re.compile(r"#\S+")
def process(line):
    hash_tmp = HASH.findall(line)
    hash_tmp_clean = []
    for hash_one in hash_tmp:
        hash_one = hash_one.lower()
        if len(hash_one) > 30:
            continue
        if hash_one[1].isalpha():
            if hash_one[-1] == '…':
                continue
            if len(hash_one) > 3 and hash_one[-3:] == '...':
                continue
            if hash_one[-1] in string.punctuation:
                hash_one = hash_one[:-1]
            hash_clean = re.findall('[a-z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                hash_tmp_clean.append(hash_one)
    return hash_tmp_clean

import json
def refine_data(fileName):
    with open('selected_hashremove_thre100_num1000_index.json', 'r', encoding='utf-8') as f:
        hash_dic = json.load(f)
    hash_refine = dict( zip(list(hash_dic.values()),list(hash_dic.keys())) )
    # hash_refine = list(hash_dic.values())
    print('refine lines')
    data_100 = []
    with open(fileName+'_clean.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            hash_tmp_clean = process(line)
            for hash_one in hash_tmp_clean:
                tmp = hash_refine.get(hash_one)
                if tmp is not None:
                    data_100.append(line.strip())
                    break
            # if len(data_100)>100000000:
            #     with open(fileName+'_clean_100.txt', 'a', encoding='utf-8') as f:
            #         for one in tqdm(data_100):
            #             f.write(one + ' \n')
            #     data_100 = []
    with open(fileName + '_clean_100.txt', 'w', encoding='utf-8') as f:
        for one in tqdm(data_100):
            f.write(one + ' \n')
    data_100 = []
def tokenization(args):
    # if args.output_dir + args.task_name is not None:
    #     os.makedirs(args.output_dir+ args.task_name, exist_ok=True)
    # Get the datasets:
    # if args.dataset_path is not None:
    # if not os.path.isfile(args.dataset_path + args.task_name +'_clean.txt'):
    #     write_data(args.dataset_path + args.task_name)
    # if not os.path.isfile(args.dataset_path + args.task_name +'_clean_100.txt'):
    #     refine_data(args.dataset_path + args.task_name)
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '.json'
    raw_datasets = datasets.load_dataset('json', data_files=data_files)
    # raw_datasets["train"] = raw_datasets["train"].shuffle()
    # Load pretrained tokenizer
    if 'bertweet' in args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer,
                                                  normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)

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
        # remove_columns=[text_column_name],
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