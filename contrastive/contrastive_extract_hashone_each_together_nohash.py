import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=1000,type=int)
parser.add_argument('--splits',default=4,type=int)
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

hash_thre_list = list(hash_dic.keys())
hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = set()
with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        hash_tmp_clean = process(line)
        for hash_one in hash_tmp_clean:
            tmp = hash_data.get(hash_one)

            if tmp is not None:
                hash_data[hash_one].add(
                    line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()
                )

import random
hash_save = []
idx = 0
hash_convert = {}
for hash_one in tqdm(hash_thre_list):
    if len(hash_data[hash_one]) < args.thre:
        continue

    if args.num > len(hash_data[hash_one]):
        for one in hash_data[hash_one]:

            hash_tmp = HASH.findall(one)
            for hash_two in hash_tmp:
                one = one.replace(hash_two, hash_two[1:].lower())

            hash_save.append({'text': one,'labels': idx})
    else:
        for one in random.sample(hash_data[hash_one], args.num):

            hash_tmp = HASH.findall(one)
            for hash_two in hash_tmp:
                one = one.replace(hash_two, hash_two[1:].lower())

            hash_save.append({'text': one,'labels': idx})
    hash_convert[idx] = hash_one
    idx+=1

# with open('./selected_thre'+str(args.thre)+'_num'+str(args.num) + '.json', 'w', encoding='utf-8') as f:
#     for one in tqdm(hash_save):
#         json.dump(one, f)
#         f.write('\n')

files = []
for tmp in range(args.splits):
    files.append(open('./selected_nohash_thre'+str(args.thre)+'_num'+str(args.num)+'_'+str(tmp)+'.json', 'w', encoding='utf-8'))

file_idx = 0
accumulate = 0
batch = int(len(hash_save)/args.splits)+1

for idx in trange(len(hash_save)):
    tmp = json.dumps(hash_save[idx], ensure_ascii=False)
    files[file_idx].write(tmp+'\n')
    accumulate+=1
    if accumulate>batch and hash_save[idx]['labels'] != hash_save[idx+1]['labels']:
        files[file_idx].close()
        file_idx+=1
        accumulate = 0
files[file_idx].close()

# with open('./selected_thre'+str(args.thre)+'_num'+str(args.num) + '.json', 'w', encoding='utf-8') as f:
#     for one in tqdm(hash_save):
#         tmp = json.dumps(one, ensure_ascii=False)
#         f.write(tmp+'\n')

with open('./selected_nohash_thre'+str(args.thre)+'_num'+str(args.num) + '_index.json', 'w', encoding='utf-8') as f:
    json.dump(hash_convert, f)