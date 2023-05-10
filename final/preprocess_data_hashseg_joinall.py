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

args = parser.parse_args()

with open(args.root+'/contrastive/hash_seg10.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    line = line.strip()
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

HASH = re.compile(r"#\S+")
def process(line):
    hash_tmp = HASH.findall(line)
    hash_tmp_clean = []
    for hash_one in hash_tmp:
        # hash_one = hash_one.lower()
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

hash_data = set()
with open(args.file, 'r', encoding='utf-8') as f:
    cur_hash = ''
    # lines = f.readlines()
    for line in tqdm(f):
        if line[:10] == 'TANS_HASH:':
            cur_hash = line.strip().split(':')[-1]
            continue
        data_tmp = line.strip()
        hash_tmp = process(data_tmp)

        if args.hashprocess == 'same':
            data_tmp = data_tmp
        elif args.hashprocess == 'remove':
            for hash_two in hash_tmp:
                data_tmp = data_tmp.replace(hash_two, '')
        elif args.hashprocess == 'seg':
            for hash_two in hash_tmp:
                tmp2 = hash_seg.get(hash_two.lower())
                if tmp2 is not None:
                    data_tmp = data_tmp.replace(hash_two, tmp2)
                else:
                    data_tmp = data_tmp.replace(hash_two, hash_two[1:])
        hash_data.add(data_tmp)

with open(args.name+'_'+args.hashprocess+'.txt', 'w', encoding='utf-8') as f:
    for one in hash_data:
        f.write(one+'\n')