import argparse
import json
import random
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='tweet_hash_clean_group.txt',type=str)
parser.add_argument('--num',default=100,type=int)

args = parser.parse_args()

with open('../contrastive/hash_his.json', 'r', encoding='utf-8') as f:
    hash_dic = json.load(f)
for hash_one in list(hash_dic.keys()):
    if hash_dic[hash_one] < args.num:
        hash_dic.pop(hash_one)
hash_thre_list = list(hash_dic.keys())
random.shuffle(hash_thre_list)

hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = []
with open(args.file, 'r', encoding='utf-8') as f:
    cur_hash = ''
    # lines = f.readlines()
    for line in f:
        if line[:10] == 'TANS_HASH:':
            cur_hash = line.strip().split(':')[-1]
            continue
        hash_data[cur_hash].append(line)

for hash_one in hash_thre_list:
    if hash_data[hash_one] < args.num:
        continue

