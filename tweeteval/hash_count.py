import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--thre', default=100, type=int)
parser.add_argument('--num', default=100, type=int)
parser.add_argument('--ran1', default=0.333, type=float)
parser.add_argument('--ran2', default=0.667, type=float)
args = parser.parse_args()

import json

f = open('../contrastive_full/thre'+str(args.thre)+'_index.json', 'r', encoding='utf-8')
hash_dic = json.load(f)
f.close()


from tqdm import tqdm, trange
import re
import string
import random

# random.seed(0)
HASH = re.compile(r"#\S+")
filePath = '/work/data/twitter_hash_clean.txt' #'twitter_hash_test_clean.txt'#


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


def write_json(fileName, data):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in tqdm(data):
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')


hash_thre_list = list(hash_dic.values())
hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = set()
hash_bad = set()
with open(filePath, 'r', encoding='utf-8') as f:
    # lines = f.readlines()
    # for idx in trange(len(lines)):
    #     line = lines[idx]
    for line in tqdm(f):
        hash_tmp_clean = process(line)
        for hash_one in hash_tmp_clean:
            tmp = hash_data.get(hash_one.lower())
            if tmp is not None:
                hash_data[hash_one.lower()].add(line)

hash_his_clean = {}
for hash_one in hash_thre_list:
    hash_his_clean[hash_one] = len(hash_data[hash_one])

with open('hash_his_clean' + '.json', 'w', encoding='utf-8') as f:
    json.dump(hash_his_clean, f)