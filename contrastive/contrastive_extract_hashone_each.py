import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=1000,type=int)
args = parser.parse_args()

import json
f=open('./hash_his.json','r',encoding='utf-8')
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
        for one in tqdm(data):
            json.dump(one, f)
            f.write('\n')

hash_thre_list = list(hash_dic.keys())
hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = set()
with open(filePath, 'r', encoding='utf-8') as f:
    # lines = f.readlines()
    # for idx in trange(len(lines)):
    #     line = lines[idx]
    for line in tqdm(f):
        hash_tmp_clean = process(line)
        for hash_one in hash_tmp_clean:
            tmp = hash_data.get(hash_one)

            if tmp is not None:
                # hash_data[hash_one].append(
                #     {'hash': hash_one, 'idx': len(hash_data[hash_one]),
                #      'text': line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]','https').strip()
                #      }
                # )
                hash_data[hash_one].add(
                    line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()
                )


hash_dic = {}
idx_save = 0
for hash_one in tqdm(hash_thre_list):
    # write_json('./'+str(args.thre)+'/'+str(idx_save), hash_data[hash_one])
    with open('./'+str(args.thre)+'/'+str(idx_save)+'.txt', 'w', encoding='utf-8') as f:
        for data in tqdm(hash_data[hash_one]):
            f.write(data+'\n')
    idx_save+=1
    hash_dic[hash_one] = idx_save

with open('./'+str(args.thre)+'/index_to_hashtag.json', 'w', encoding='utf-8') as f:
    json.dump(hash_dic, f)