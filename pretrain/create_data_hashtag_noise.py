import argparse
import json
import random
from tqdm import tqdm,trange
import numpy as np
import re
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='tweet_hash_clean_group.txt',type=str)
parser.add_argument('--num',default=100,type=int)
parser.add_argument('--name',default='tweet_hash_clean_group_raw',type=int)
parser.add_argument('--ran1', default=0.333, type=float)
parser.add_argument('--ran2', default=0.667, type=float)

args = parser.parse_args()

with open('../contrastive/hash_his.json', 'r', encoding='utf-8') as f:
    hash_dic = json.load(f)
for hash_one in list(hash_dic.keys()):
    if hash_dic[hash_one] < args.num:
        hash_dic.pop(hash_one)
hash_thre_list = list(hash_dic.keys())
random.shuffle(hash_thre_list)

with open('../contrastive/hash_seg.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

HASH = re.compile(r"#\S+")

hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = []
with open(args.file, 'r', encoding='utf-8') as f:
    cur_hash = ''
    # lines = f.readlines()
    for line in tqdm(f):
        if line[:10] == 'TANS_HASH:':
            cur_hash = line.strip().split(':')[-1]
            continue
        hash_data[cur_hash].append(line)

for hash_one in hash_thre_list:
    if hash_data[hash_one] < args.num:
        continue
    with open(args.name+'_'+str(args.num)+'.txt', 'a', encoding='utf-8') as f:
        for data_tmp in hash_data[hash_one]:

            ran1 = np.random.random()
            if ran1 < args.ran1:
                hash_tmp = HASH.findall(data_tmp)
                for hash_two in hash_tmp:
                    data_tmp = data_tmp.replace(hash_two, '')
            elif ran1 < args.ran2:
                hash_tmp = HASH.findall(data_tmp)
                for hash_two in hash_tmp:
                    tmp2 = hash_seg.get(hash_two.lower())
                    if tmp2 is not None:
                        data_tmp = data_tmp.replace(hash_two, tmp2)
                    else:
                        data_tmp = data_tmp.replace(hash_two, hash_two[1:])

            f.write(data_tmp)
