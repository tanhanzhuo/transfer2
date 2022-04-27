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
        if len(hash_tmp_clean) == 1:
            hash_one = hash_tmp_clean[0]
            if hash_one in hash_dic.keys():
                hash_dic[hash_one] += 1
            else:
                hash_dic[hash_one] = 1

with open('hash_his_one.txt', 'w') as f:
    for hash_one in hash_dic.keys():
        if hash_dic[hash_one] > 5:
            f.write(hash_one + '\t' + str(hash_dic[hash_one]) + '\n')