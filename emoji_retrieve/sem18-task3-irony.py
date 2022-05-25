# keywords = ['irony', 'sarcasm', 'not']

import argparse
import os
from tqdm import tqdm, trange
import random
import re
import json
import string

parser = argparse.ArgumentParser()
parser.add_argument('--hash', default='#irony,#sarcasm,#not',type=str)
parser.add_argument('--num', default=1000000,type=int)
parser.add_argument('--name', default='sem18-task3-irony',type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    SELECT_NUM = args.num

    # with open('../hashtag/hash_his.txt','r',encoding='utf-8') as f:
    #     lines = f.readlines()
    # hash_dic = {}
    # for line in lines:
    #     line = line.strip()
    #     hashtag = line.split('\t')[0]
    #     num = int(line.split('\t')[1])
    #     if hashtag in args.hash.split(','):
    #         hash_dic[hashtag] = num
    # print(hash_dic)

    HASH = re.compile(r"#\S+")
    filePath = '/work/data/twitter_hash.txt'
    with open(filePath, 'r') as f:
        lines = f.readlines()
    # random.shuffle(lines)

    hash_dic = {}
    for one in args.hash.split(','):
        hash_dic[one] = 0
    data_hash = []
    for line in tqdm(lines):
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
        flag = 0
        for hash_one in hash_tmp_clean:
            if hash_one in args.hash.split(','):
                if hash_dic[hash_one] < SELECT_NUM:
                    flag = 1
                    break
        if flag:
            hash_dic[hash_one] += 1
            data_line = line
            for tmp in hash_tmp:
                data_line = data_line.replace(tmp, '')
            data_hash.append(hash_one + '\t' + data_line)

    with open('data_task_' + str(args.name) + '_' + str(SELECT_NUM) +'.txt', 'w') as f:
        for data in data_hash:
            f.write(data)
