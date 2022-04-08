import re
from nltk.tokenize import wordpunct_tokenize
HASH = re.compile(r"#\S+")
def read_data(fileName):
    data = []
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.split('\t')[1])
    return data

hash_dic = {}
for task in ['stance/face_masks','stance/fauci','stance/school_closures','stance/stay_at_home_orders','hate','sem-17','sem-18']:
    input_dir = '../finetune/data/' + task + '/'
    for SP in ['train','dev','test']:
        data = read_data(input_dir+SP)
        for line in data:
            hash_tmp = HASH.findall(line)
            for hash_one in hash_tmp:
                hash_sep = wordpunct_tokenize(hash_one)
                if hash_sep[0] == '#':
                    hash_one = hash_sep[1].lower()
                    if hash_one in hash_dic.keys():
                        continue
                    else:
                        hash_dic[hash_one] = 0




import os
from tqdm import tqdm, trange
import numpy as np
import random

filePath = '/work/data/twitter_hash.txt'
with open(filePath, 'r') as f:
    lines = f.readlines()
    print(len(lines))

data = []
for idx in trange(len(lines)):
    line = lines[idx]
    if line.strip():
        hash_tmp = HASH.findall(line)
        for hash_one in hash_tmp:
            hash_sep = wordpunct_tokenize(hash_one)
            if hash_sep[0] == '#':
                hash_one = hash_sep[1].lower()
                if hash_one in hash_dic.keys():
                    if hash_dic[hash_one] < 100:
                        data.append(line)
                        hash_dic[hash_one]+=1

with open('data_extension.txt', 'w') as f:
    for line in data:
        f.write(line)