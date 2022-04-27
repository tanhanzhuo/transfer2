with open('hash_his.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

hash_dic = {}
for line in lines:
    line = line.strip()
    hashtag = line.split('\t')[0]
    num = int(line.split('\t')[1])
    if num > 10000:
        hash_dic[hashtag] = num

hash_dic_sort = dict(sorted(hash_dic.items(), key=lambda x: x[1], reverse=True))
SELECT = list(hash_dic_sort.keys())[:100]


import os
from tqdm import tqdm, trange
import re
import json
import string
HASH = re.compile(r"#\S+")
hash_dic = {}
filePath = '/work/data/twitter_hash.txt'
with open(filePath, 'r') as f:
    for line in tqdm(f):
        if not line:
            continue
        hash_tmp = HASH.findall(line)
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
                    if hash_one in hash_dic.keys():
                        hash_dic[hash_one] += 1
                    else:
                        hash_dic[hash_one] = 1
