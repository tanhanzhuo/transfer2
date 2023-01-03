import json
import argparse
import numpy as np
import os
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='eval-irony,eval-hate,eval-offensive,eval-emotion,eval-stance',type=str)
parser.add_argument('--input',default='.json',type=str)
parser.add_argument('--output',default='_en.txt',type=str)
args = parser.parse_args()

if args.input.split('.')[-1] == 'json' and args.output.split('.')[-1] == 'txt':
    for dataset_one in args.dataset.split(','):
        for sp in ['train','dev','test']:
            data_sem = []
            with open('../finetune/data/' + dataset_one + '/' + sp + '.json', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    one = json.loads(line)
                    data_sem.append(one)
            with open('../finetune/data/' + dataset_one + '/' + sp + args.output.split('.')[0], 'w', encoding='utf-8') as f:
                for line in data_sem:
                    f.write(line['text']+'\n')
#
# elif args.input.split('.')[-1] == 'txt':
#     for dataset_one in args.dataset.split(','):
#         print('loading data')
#         data_sem = []
#         for sp in ['train','dev','test']:
#             with open('../finetune/data/' + dataset_one + '/' + sp + '.json', 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     one = json.loads(line)
#                     data_sem.append(one)