import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--thre', default=100, type=int)
parser.add_argument('--num', default=100, type=int)
parser.add_argument('--ran1', default=0.333, type=float)
parser.add_argument('--ran2', default=0.667, type=float)
args = parser.parse_args()

import json

f = open('./thre100_index.json', 'r', encoding='utf-8')
hash_dic = json.load(f)
f.close()

with open('../contrastive/hash_seg.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

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

NUM = args.num
hash_pair = []
hash_idx = 0
for hash_one in tqdm(hash_thre_list):
    data = list(hash_data[hash_one])
    # if len(data) < args.thre:
    #     continue

    for tmp in range(NUM):
        data_tmp = random.sample(data, 2)
        # hash_pair.append(  {'text1':lines[data_tmp[0]].replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip(), \
        #                     'text2':lines[data_tmp[1]].replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()}  )
        ran1 = np.random.random()
        if  ran1 < 0.333:
            hash_tmp = HASH.findall(data_tmp[0])
            for hash_two in hash_tmp:
                data_tmp[0] = data_tmp[0].replace(hash_two, '')
        elif ran1 < 0.667:
            hash_tmp = HASH.findall(data_tmp[0])
            for hash_two in hash_tmp:
                data_tmp[0] = data_tmp[0].replace(hash_two, hash_two[1:])
        else:
            hash_tmp = HASH.findall(data_tmp[0])
            for hash_two in hash_tmp:
                tmp2 = hash_seg.get(hash_two.lower())
                if tmp2 is not None:
                    data_tmp[0] = data_tmp[0].replace(hash_two, tmp2)
                else:
                    data_tmp[0] = data_tmp[0].replace(hash_two, hash_two[1:])


        ran1 = np.random.random()
        if  ran1 < args.ran1:
            hash_tmp = HASH.findall(data_tmp[1])
            for hash_two in hash_tmp:
                data_tmp[1] = data_tmp[1].replace(hash_two, '')
        elif ran1 < args.ran1:
            hash_tmp = HASH.findall(data_tmp[1])
            for hash_two in hash_tmp:
                data_tmp[1] = data_tmp[1].replace(hash_two, hash_two[1:])
        else:
            hash_tmp = HASH.findall(data_tmp[1])
            for hash_two in hash_tmp:
                tmp2 = hash_seg.get(hash_two.lower())
                if tmp2 is not None:
                    data_tmp[1] = data_tmp[1].replace(hash_two, tmp2)
                else:
                    data_tmp[1] = data_tmp[1].replace(hash_two, hash_two[1:])
        hash_pair.append({'text1': data_tmp[0], 'text2': data_tmp[1], 'label':hash_idx})
    hash_idx += 1
random.shuffle(hash_pair)
write_json('hash_pairmix_thre' + str(args.thre) + '_num' + str(args.num), hash_pair)