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
parser.add_argument('--method_hash',default='modelT100N100R_fileT100N100R_num10_top20_textfirst',type=str)
parser.add_argument('--top',default=3,type=int)
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


    for sp in ['train', 'dev', 'test']:
        train_dataset = raw_datasets[sp]
        text_hash = raw_datasets_hash[sp]['text']
        save_hash = []
        for idx_tmp in range(args.top):
            save_hash.append([])
        for idx in range(len(text_hash)):
            all_hash = text_hash[idx].split(' \n ')
            if len(all_hash)!= 22:
                print('error!!!!!!!!!!!!!!!! not same number of top emoji/hash')
                print(args.task_name+'_'+sp)
                print(train_dataset['text'][idx])
                raise ValueError('not same number of top emoji/hash')
            for idx_tmp in range(args.top):
                save_hash[idx_tmp].append(all_hash[idx_tmp].strip())
        for idx_tmp in range(args.top):
            train_dataset = train_dataset.add_column("hash"+str(idx_tmp), save_hash[idx_tmp])

        raw_datasets[sp] = train_dataset.shuffle()

    # Load pretrained tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,normalization=True)

    # First we tokenize all the texts.

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = False

    def tokenize_function(examples):
        total = len(examples['text'])
        sentences = examples['text']
        for idx_tmp in range(args.top):
            sentences = sentences + examples['hash'+str(idx_tmp)]
        sent_features = tokenizer(
            sentences,
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=False,
        )
        features = {}
        if args.top == 1:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total]] for i in
                    range(total)]
        elif args.top == 2:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2] ] for i in
                    range(total)]
        elif args.top == 3:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2],
                     sent_features[key][i + total * 3]] for i in
                    range(total)]
        elif args.top == 4:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2],
                     sent_features[key][i + total * 3], sent_features[key][i + total * 4]] for i in
                    range(total)]
        elif args.top == 5:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2],
                     sent_features[key][i + total * 3], sent_features[key][i + total * 4],
                     sent_features[key][i + total * 5]] for i in
                    range(total)]
        else:
            print('error: wrong top K')


        features['labels'] = examples['labels']
        return features

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,#############need test
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

        save_hash = args_tmp.method_hash.split('top')[0]
        tokenized_datasets.save_to_disk(args_tmp.dataset_path + args_tmp.task_name + '/hash_' \
                                        + save_hash + 'top_'+str(args_tmp.top))