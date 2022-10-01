import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import random

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='twitter_hash_sep_thre100_num100',type=str)
parser.add_argument("--NUM_SPLIT", default=4, type=int)
parser.add_argument("--TOP", default=5, type=int)
args = parser.parse_args()


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
            hash_clean = re.findall('[a-zA-Z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                hash_tmp_clean.append(hash_one)
    return hash_tmp_clean

with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)
hashtags = list(CONVERT.values())

data_all = {}
hashtags_check = {}
for hash_one in hashtags:
    data_all[hash_one] = []
    hashtags_check[hash_one] = []
for sp in range(args.NUM_SPLIT):
    with open(args.file + '_' + str(sp) + '.json', 'r', encoding='utf-8') as f:
        for line in f:
            data_one = json.loads(line)
            data_all[CONVERT[str(data_one['labels'])]].append(data_one['text'])

file = open('hash_word_relation.txt', 'a', encoding='utf-8')
for hash_idx in trange(len(hashtags)):
    hash_one = hashtags[hash_idx]
    data_one = data_all[hash_one]
    hash_relation = {}
    for text in data_one:
        hash_tmp = process(text)
        for hash_tmp_one in hash_tmp:
            tmp = hashtags_check.get(hash_tmp_one)
            if tmp is None:
                continue
            if hash_tmp_one in hash_relation.keys():
                hash_relation[hash_tmp_one] += 1
            else:
                hash_relation[hash_tmp_one] = 1
    hash_sort = dict(sorted(hash_relation.items(), key=lambda x: x[1], reverse=True))
    if len(hash_sort.keys()) == 1:
        val = []
    else:
        val = list(hash_sort.keys())[1:args.TOP+1]
    if len(hash_sort.keys()) < args.TOP+1:
        val_tmp = random.sample(hashtags, args.TOP+1-len(hash_sort.keys()))
        for tmp in val_tmp:
            val.append(tmp)

    tmp_merge = '' + hash_one
    for hash_tmp in val:
        tmp_merge = tmp_merge + '\t' + hash_tmp
    tmp_merge += ' \n'
    file.write(tmp_merge)
file.close()