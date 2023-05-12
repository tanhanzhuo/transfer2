import argparse
import json
import random
from tqdm import tqdm,trange
import numpy as np
import re
import string

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='tweet_hash_clean_group.txt',type=str)
parser.add_argument('--name',default='tweet_hash_clean',type=str)
parser.add_argument('--root', default='..', type=str)
parser.add_argument('--hashprocess', default='same', type=str)
parser.add_argument('--thre', default=1000, type=int)

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
        hash_data[cur_hash].append(data_tmp)

print('number of hashtags:{}'.format(len(hash_data.keys())))
low = 0
hash_data_high = set()
for hash_one in hash_data.keys():
    hash_data_one = hash_data[hash_one]
    if len(hash_data_one) < args.thre:
        low += 1
        continue
    else:
        for sample in hash_data_one:
            hash_data_high.add(sample)
print('number of lower hashtags:{}'.format(low))
print('number of final samples:{}'.format(len(hash_data_high)))