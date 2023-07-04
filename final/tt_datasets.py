# import datasets
# from datasets import concatenate_datasets, load_dataset
# data_all = datasets.load_from_disk('../finetune/data/eval-stance/token')
# data_2 = concatenate_datasets([data_all['train'],data_all['dev']])
# data_3 = data_2.train_test_split(test_size=0.1)
#
# from pathlib import Path
# import csv
# path = Path('../finetune/data/eval-stance/train_same_500_one20_top100_sp.tsv')
#
# def load_tsv(fname):
#     with open(fname, 'r',encoding='utf-8') as f:
#         reader = csv.DictReader(f, delimiter='\t')
#         for row in reader:
#             yield row
#
# for x in load_tsv(path):
#     print(x)



import argparse
import json
import random
from tqdm import tqdm,trange
import numpy as np
import re
import string

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='../pretrain/hashtag/tweet_hash_clean_group.txt',type=str)
parser.add_argument('--name',default='tweet_hash_clean',type=str)
parser.add_argument('--root', default='..', type=str)
parser.add_argument('--hashprocess', default='same', type=str)
parser.add_argument('--num', default=500, type=int)
parser.add_argument('--thre', default=100, type=int)

args = parser.parse_args()

hash_data = {}
with open(args.file, 'r', encoding='utf-8') as f:
    cur_hash = ''
    # lines = f.readlines()
    for line in tqdm(f):
        if line[:10] == 'TANS_HASH:':
            cur_hash = line.strip().split(':')[-1]
            hash_data[cur_hash] = []
            continue
        data_tmp = line.strip()
        if args.hashprocess == 'same':
            data_tmp = data_tmp
        hash_data[cur_hash].append(data_tmp)

for hash_one in list(hash_data.keys()):
    hash_data_one = hash_data[hash_one]
    if len(hash_data_one) < args.thre:
        hash_data.pop(hash_one)

print(len(hash_data.keys()))