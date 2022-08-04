import argparse
import emoji

parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=1000,type=int)
parser.add_argument('--splits',default=4,type=int)
args = parser.parse_args()

import json
f=open('./emoji_num.json','r',encoding='utf-8')
emoji_dic = json.load(f)
f.close()
for emoji_one in list(emoji_dic.keys()):
    if emoji_dic[emoji_one] < args.thre:
        emoji_dic.pop(emoji_one)

from tqdm import tqdm, trange
import re
import string
import random

filePath = '/work/data/twitter_emoji.txt'#'twitter_emoji_sample.txt'

def process(line):
    emoji_hash = ''
    line = line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip()
    if len(line.replace(' ', '')) < 10:
        return ''
    emoji_list = emoji.distinct_emoji_list(line)
    if len(emoji_list) > 0:
        emoji_hash = ''.join(emoji_list)

    return emoji_hash

emoji_thre_list = list(emoji_dic.keys())
emoji_data = {}
for emoji_one in emoji_thre_list:
    emoji_data[emoji_one] = set()

with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        line = line.split('\t')[-1]
        emoji_tmp_clean = process(line)
        if len(emoji_tmp_clean) < 1:
            continue
        tmp = emoji_data.get(emoji_tmp_clean)
        if tmp is not None:
            line = emoji.replace_emoji( line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip() )
            emoji_data[emoji_tmp_clean].add(line)

import random
emoji_save = []
idx = 0
emoji_convert = {}
for emoji_one in tqdm(emoji_thre_list):
    if len(emoji_data[emoji_one]) < args.thre:
        continue

    if args.num > len(emoji_data[emoji_one]):
        for one in emoji_data[emoji_one]:
            emoji_save.append({'text': one,'labels': idx})
    else:
        # for one in random.sample(emoji_data[emoji_one], args.num):
        data_tmp = list(emoji_data[emoji_one])
        idx_tmp = list(range(len(data_tmp)))
        random.shuffle(idx_tmp)
        for tmp in idx_tmp[:args.num]:
            one = data_tmp[tmp]
            emoji_save.append({'text': one,'labels': idx})
    emoji_convert[idx] = emoji_one
    idx+=1

# with open('./selected_thre'+str(args.thre)+'_num'+str(args.num) + '.json', 'w', encoding='utf-8') as f:
#     for one in tqdm(emoji_save):
#         json.dump(one, f)
#         f.write('\n')

files = []
for tmp in range(args.splits):
    files.append(open('./selected_noemoji_thre'+str(args.thre)+'_num'+str(args.num)+'_'+str(tmp)+'.json', 'w', encoding='utf-8'))

file_idx = 0
accumulate = 0
batch = int(len(emoji_save)/args.splits)+1

for idx in trange(len(emoji_save)):
    tmp = json.dumps(emoji_save[idx], ensure_ascii=False)
    files[file_idx].write(tmp+'\n')
    accumulate+=1
    if accumulate>batch and emoji_save[idx]['labels'] != emoji_save[idx+1]['labels']:
        files[file_idx].close()
        file_idx+=1
        accumulate = 0
files[file_idx].close()

# with open('./selected_thre'+str(args.thre)+'_num'+str(args.num) + '.json', 'w', encoding='utf-8') as f:
#     for one in tqdm(emoji_save):
#         tmp = json.dumps(one, ensure_ascii=False)
#         f.write(tmp+'\n')

with open('./selected_noemoji_thre'+str(args.thre)+'_num'+str(args.num) + '_index.json', 'w', encoding='utf-8') as f:
    json.dump(emoji_convert, f)
idx = 0
emoji_num = {}
for emoji_one in tqdm(emoji_thre_list):
    if len(emoji_data[emoji_one]) < args.thre:
        continue

    emoji_num[idx] = len(emoji_data[emoji_one])
    idx+=1
with open('./selected_noemoji_thre'+str(args.thre)+'_num'+str(args.num) + '_num.json', 'w', encoding='utf-8') as f:
    json.dump(emoji_convert, f)