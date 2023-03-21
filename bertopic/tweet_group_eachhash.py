import argparse
import json
import random

import torch
from tqdm import tqdm,trange
import numpy as np
import re
import string
from bertopic import BERTopic
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='../pretrain/hashtag/tweet_hash_clean_group_all.txt',type=str)
parser.add_argument('--num',default=100,type=int)
parser.add_argument('--name',default='tweet_hash_clean_group_raw',type=str)

args = parser.parse_args()

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

def write_json(fileName, data):
    with open(fileName + '.json', 'a', encoding='utf-8') as f:
        for one in tqdm(data):
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')

with open('../contrastive/hash_his.json', 'r', encoding='utf-8') as f:
    hash_dic = json.load(f)

hash_thre_list = list(hash_dic.keys())
# random.shuffle(hash_thre_list)

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

# embedding_model = pipeline("feature-extraction", model="princeton-nlp/sup-simcse-roberta-base", device=0)
embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
topic_model = BERTopic(embedding_model=embedding_model, verbose=False)


for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < args.num:
        hash_data.pop(hash_one)

hash_data_group = []
for hash_one in tqdm(hash_data.keys()):
    if len(hash_data[hash_one]) < args.num:
        continue
    hash_data_one = hash_data[hash_one]
    random.shuffle(hash_data_one)

    hash_data_two = []
    for data_tmp in hash_data_one:
        data_tmp = data_tmp.replace('@USER','').replace('https','')
        hash_tmp = process(data_tmp)
        for hash_two in hash_tmp:
            data_tmp = data_tmp.replace(hash_two, '')
        hash_data_two.append(data_tmp)
    with torch.no_grad():
        topics, probs = topic_model.fit_transform(hash_data_two)
    hash_data_one_group = {'hashtag':hash_one}
    for idx in range(len(hash_data_one)):
        if topics[idx] + 1 in hash_data_one_group.keys():
            hash_data_one_group[topics[idx] + 1].append(hash_data_one[idx])
        else:
            hash_data_one_group[topics[idx] + 1] = [hash_data_one[idx]]

    hash_data_group.append(hash_data_one_group)
    if len(hash_data_group) > 100:
        write_json(args.name + '_' + str(args.num), hash_data_group)
        hash_data_group = []


