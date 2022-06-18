import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=1000,type=int)
args = parser.parse_args()

import json
f=open('./hash_his.json','r',encoding='utf-8')
hash_dic = json.load(f)
f.close()

from tqdm import tqdm, trange
import re
import string
import random
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

hash_data = {}
with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        hash_tmp = process(line)
        for hash_one in hash_tmp:
            if hash_dic[hash_one] >= args.thre:
                if hash_one in hash_data.keys():
                    hash_data[hash_one].append(line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip())
                else:
                    hash_data[hash_one] = [line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip()]

NUM = args.num
hash_pair = []
for hash_one in hash_data.keys():
    data = hash_data[hash_one]
    for tmp in range(NUM):
        data_tmp = random.sample(data, 2)
        hash_pair.append(  {'text1':data_tmp[0], 'text2':data_tmp[1]}  )
write_json('hash_pair_thre'+str(args.thre)+'_num'+str(args.num), hash_pair)