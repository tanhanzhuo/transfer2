import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=1000,type=int)
args = parser.parse_args()

import json
f=open('../contrastive/hash_his.json','r',encoding='utf-8')
hash_dic = json.load(f)
f.close()
for hash_one in list(hash_dic.keys()):
    if hash_dic[hash_one] < args.thre:
        hash_dic.pop(hash_one)

from tqdm import tqdm, trange
import re
import string
import random
random.seed(0)
HASH = re.compile(r"#\S+")
filePath = '/work/data/twitter_hash_clean.txt'#'twitter_hash_sample.txt'

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
            hash_clean = re.findall('[a-z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                hash_tmp_clean.append(hash_one)

    return hash_tmp_clean

hash_thre_list = list(hash_dic.keys())
hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = []
with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        hash_tmp_clean = process(line)
        for hash_one in hash_tmp_clean:
            tmp = hash_data.get(hash_one)
            if tmp is not None:
                hash_data[hash_one].append(line.strip())

import random
idx = 0
hash_convert = {}
for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < 100:
        continue
    hash_convert[idx] = hash_one
    idx+=1
with open('./thre100_index.json', 'w', encoding='utf-8') as f:
    json.dump(hash_convert, f)

idx = 0
hash_convert = {}
for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < 1000:
        continue
    hash_convert[idx] = hash_one
    idx+=1
with open('./thre1000_index.json', 'w', encoding='utf-8') as f:
    json.dump(hash_convert, f)

import random
hash_save = set()
THRE = 100
NUM = 1000
for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < THRE:
        continue
    if len(hash_data[hash_one]) < NUM:
        for one in hash_data[hash_one]:
            hash_save.add(one)
    else:
        # for one in random.sample(hash_data[hash_one], args.num):
        data_tmp = list(hash_data[hash_one])
        idx_tmp = list(range(len(data_tmp)))
        random.shuffle(idx_tmp)
        for tmp in idx_tmp[:NUM]:
            one = data_tmp[tmp]
            hash_save.add(one)

hash_save = list(hash_save)
with open('twitter_hash_thre100_num1000.json', 'w', encoding='utf-8') as f:
    for idx in trange(len(hash_save)):
        tmp = json.dumps({'text':hash_save[idx],'labels':idx}, ensure_ascii=False)
        f.write(tmp + '\n')
    # for line in hash_save:
    #     f.write(line + ' \n')

hash_save = set()
THRE = 1000
NUM = 1000
for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < THRE:
        continue
    if len(hash_data[hash_one]) < NUM:
        for one in hash_data[hash_one]:
            hash_save.add(one)
    else:
        # for one in random.sample(hash_data[hash_one], args.num):
        data_tmp = list(hash_data[hash_one])
        idx_tmp = list(range(len(data_tmp)))
        random.shuffle(idx_tmp)
        for tmp in idx_tmp[:NUM]:
            one = data_tmp[tmp]
            hash_save.add(one)

hash_save = list(hash_save)
with open('twitter_hash_thre1000_num1000.json', 'w', encoding='utf-8') as f:
    for idx in trange(len(hash_save)):
        tmp = json.dumps({'text':hash_save[idx],'labels':idx}, ensure_ascii=False)
        f.write(tmp + '\n')
    # for line in hash_save:
    #     f.write(line + ' \n')