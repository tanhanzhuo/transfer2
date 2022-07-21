import argparse
import os
import datasets
import json
from transformers import  AutoTokenizer
import copy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='/work/transfer2/finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='/work/transfer2/finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method',default='modelT100N100S_fileT100S_num10_cluster_top0_hashlast',type=str)
parser.add_argument("--tokenizer_name", default='vinai/bertweet-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")
parser.add_argument('--hash_file',default='feature_modelT100N100S_fileT100S_num10_cluster',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory

hash_samples = []
hash_embs = []
hash_tags = []
for idx in range(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    # hash_embs.extend(tmp['center_embs'])
    hash_embs.extend(tmp['center_embs'])
    hash_tags.extend(tmp['center_hash'])
    tmp.close()


def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append({'labels': line.split('\t')[0], 'text': line.split('\t')[1]})
    return data

def write_json(fileName):
    data = read_data(fileName)
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')

def tokenization(args):
    # if args.output_dir + args.task_name is not None:
    #     os.makedirs(args.output_dir+ args.task_name, exist_ok=True)
    # Get the datasets:
    # if args.dataset_path is not None:
    # if not os.path.isfile(args.dataset_path + args.task_name +'/train.json'):
    for fileName in ['train', 'dev', 'test']:
        write_json(args.dataset_path + args.task_name + '/' + fileName)
    data_files = {}
    data_files["train"] = args.dataset_path + args.task_name + '/train.json'
    data_files["dev"] = args.dataset_path + args.task_name + '/dev.json'
    data_files["test"] = args.dataset_path + args.task_name + '/test.json'
    raw_datasets = datasets.load_dataset('json', data_files=data_files)
    raw_datasets["train"] = raw_datasets["train"].shuffle()
    raw_datasets["dev"] = raw_datasets["dev"].shuffle()
    raw_datasets["test"] = raw_datasets["test"].shuffle()
    # Load pretrained tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer,
                                                  normalization=True)

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
            return_special_tokens_mask=True,
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
    for task in args.task_name.split(','):
        args_tmp.task_name = task
        tokenized_datasets = tokenization(args_tmp)
        tokenized_datasets.save_to_disk(args_tmp.dataset_path + args_tmp.task_name + '/token')