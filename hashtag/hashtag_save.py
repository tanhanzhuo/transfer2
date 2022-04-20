import os
from tqdm import tqdm, trange
import re
import json
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
                if hash_one in hash_dic.keys():
                    hash_dic[hash_one] += 1
                else:
                    hash_dic[hash_one] = 1
hash_dic_clean = {}
for hash_one in hash_dic.keys():
    if hash_dic[hash_one] > 5:
        hash_dic_clean[hash_one] = hash_dic[hash_one]


with open('hash_his.json', 'w') as f:
    for one in hash_dic_clean:
        json.dump(one, f)
        f.write('\n')
