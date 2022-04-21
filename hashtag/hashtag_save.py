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

with open('hash_his.txt', 'w') as f:
    for hash_one in hash_dic.keys():
        if hash_dic[hash_one] > 5:
            f.write(hash_one + '\t' + str(hash_dic[hash_one]) + '\n')
