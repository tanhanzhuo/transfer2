import argparse
import json
import random
from tqdm import tqdm,trange
import numpy as np
import re
import string

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='tweet_hash_clean_group_all.txt',type=str)
parser.add_argument('--num',default=20,type=int)
parser.add_argument('--name',default='tweet_hash_clean_group_raw',type=str)
parser.add_argument('--ran1', default=0.333, type=float)
parser.add_argument('--ran2', default=0.667, type=float)
parser.add_argument('--max_len', default=512, type=int)
parser.add_argument('--rep', default=5, type=int)
parser.add_argument('--balance', default=0, type=int)
parser.add_argument('--root', default='../../', type=str)
parser.add_argument('--sep', default=0, type=int)

args = parser.parse_args()

with open(args.root+'/contrastive/hash_his.json', 'r', encoding='utf-8') as f:
    hash_dic = json.load(f)
# for hash_one in list(hash_dic.keys()):
#     if hash_dic[hash_one] < args.num:
#         hash_dic.pop(hash_one)
hash_thre_list = list(hash_dic.keys())
random.shuffle(hash_thre_list)

with open(args.root+'/contrastive/hash_seg10.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    line = line.strip()
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

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
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
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

hash_pair = []
hash_idx = 0
for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < args.num:
        continue
    if args.balance == 0:
        epoch = args.rep
    else:
        epoch = min(args.rep, int( args.balance*args.rep*1.0/len(hash_data[hash_one]) ) )
    for tmp in range(epoch):
        hash_data_one = hash_data[hash_one]
        random.shuffle(hash_data_one)
        hash_data_one_noise = []
        for data_tmp in hash_data_one:
            data_tmp = data_tmp.strip()
            hash_tmp = process(data_tmp)
            for hash_two in hash_tmp:
                ran1 = np.random.random()
                if ran1 < args.ran1:
                    data_tmp = data_tmp.replace(hash_two, '')
                elif ran1 < args.ran2:
                    tmp2 = hash_seg.get(hash_two.lower())
                    if tmp2 is not None:
                        data_tmp = data_tmp.replace(hash_two, tmp2)
                    else:
                        data_tmp = data_tmp.replace(hash_two, hash_two[1:])

            hash_data_one_noise.append(data_tmp)
        hash_data_group = ''
        for idx in range(len(hash_data_one_noise)):
            if args.sep == 0:
                hash_data_group += hash_data_one_noise[idx] + ' '
            else:
                hash_data_group += hash_data_one_noise[idx] + ' </s> '
            if len(hash_data_group) > args.max_len*0.95:
                rand_idx = int(random.rand() * len(hash_data_one_noise))
                hash_pair.append({'text1': hash_data_group, 'text2': hash_data_one_noise[rand_idx], 'label':hash_idx})
                hash_data_group = ''
    hash_idx += 1


        # with open(args.name + '_' + str(args.num) + '.txt', 'a', encoding='utf-8') as f:
        #     hash_data_group = ''
        #     for idx in range(len(hash_data_one_noise)):
        #         if args.sep == 0:
        #             hash_data_group += hash_data_one_noise[idx] + ' '
        #         else:
        #             hash_data_group += hash_data_one_noise[idx] + ' </s> '
        #         if len(hash_data_group) > args.max_len*0.95:
        #             f.write(hash_data_group+'\n')
        #             hash_data_group = ''
write_json(args.name + '_' + str(args.num), hash_pair)
