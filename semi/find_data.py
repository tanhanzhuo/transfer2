import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import os
import copy
import random
import re
from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='../pretrain/hashtag/tweet_hash_clean_group_all.txt',type=str)
args = parser.parse_args()

with open('../contrastive/hash_his.json', 'r', encoding='utf-8') as f:
    hash_dic = json.load(f)
# for hash_one in list(hash_dic.keys()):
#     if hash_dic[hash_one] < args.num:
#         hash_dic.pop(hash_one)
hash_thre_list = list(hash_dic.keys())
random.shuffle(hash_thre_list)

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


for hash_one in hash_thre_list[0:2]:
    hash_data_one = hash_data[hash_one]
    hash_data_two = []
    for data_tmp in hash_data_one:
        hash_tmp = process(data_tmp)
        for hash_two in hash_tmp:
            data_tmp = data_tmp.replace(hash_two, '')
        hash_data_two.append(data_tmp)

    inputs = tokenizer(hash_data_two, truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor(inputs['input_ids']).cuda(),
                        attention_mask=torch.tensor(inputs['attention_mask']).cuda(),
                        # token_type_ids=torch.tensor([inputs['token_type_ids']]).cuda(),
                        output_hidden_states=True, return_dict=True).pooler_output
        dis = cos_sim(outputs[-1], outputs[:-1])
        val, best_idx = dis.topk(len(dis))
    with open(hash_one+'.txt', 'w', encoding='utf-8') as f:
        f.write(hash_data_one[-1])
        for idx in best_idx:
            f.write(hash_data_one[idx])
