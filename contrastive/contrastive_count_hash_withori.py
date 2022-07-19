import os
import time
from multiprocessing import Pool
from tqdm import tqdm, trange
import re
import json
import string
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pool',default=20,type=int)
parser.add_argument('--num',default=1000,type=int)

import numpy as np
hash_tags = []
for idx in range(4):
    tmp = np.load('feature_modelT100N100S_fileT100S_num1'+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_tags.extend(tmp['center_hash'])
    tmp.close()

hash_dic = {}
for one in hash_tags:
    hash_dic[one] = {}

HASH = re.compile(r"#\S+")
filePath = '/work/data/twitter_hash.txt'#'twitter_hash_sample.txt'

def process(line):
    line = line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip()
    if len(line) < 10:
        return []
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

def write_json(fileName,data):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

if __name__ == "__main__":
    args = parser.parse_args()

    with open(filePath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip()
            if len(line) < 10:
                continue
            hash_tmp = HASH.findall(line)
            hash_tmp_clean = []
            for hash_one in hash_tmp:
                hash_up = hash_one
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
                        tmp = hash_dic.get(hash_one)
                        if tmp is not None:
                            if hash_up in hash_dic[hash_one].keys():
                                hash_dic[hash_one][hash_up] += 1
                            else:
                                hash_dic[hash_one][hash_up] = 1

    # time3 = time.time()
    # print(time3-time2)
    hash_convert = {}
    cur_num = 0
    for hash_one in list(hash_dic.keys()):
        hash_up = ''
        hash_num = 0
        if cur_num % 1000 ==0:
            print(hash_dic[hash_one])
        for hash_tmp in hash_dic[hash_one].keys():
            if hash_dic[hash_one][hash_tmp] > hash_num:
                hash_up = hash_tmp
        if hash_up != '':
            hash_convert[hash_one] = hash_up
            cur_num+=1
        else:
            print('*********************error**********************')
            print(hash_one)
            print(hash_dic[hash_one])


    with open('hash_convert' + '.json', 'w', encoding='utf-8') as f:
        json.dump(hash_convert, f)