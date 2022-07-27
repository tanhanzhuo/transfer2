import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=1000,type=int)
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
# random.seed(0)
import emoji
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

def write_json(fileName,data):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in tqdm(data):
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

emoji_thre_list = list(emoji_dic.keys())
emoji_data = {}
for emoji_one in emoji_thre_list:
    emoji_data[emoji_one] = set()

with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        emoji_tmp_clean = process(line)
        if len(emoji_tmp_clean) < 1:
            continue
        tmp = emoji_data.get(emoji_tmp_clean)
        if tmp is not None:
            line = emoji.replace_emoji( line.replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip() )
            emoji_data[emoji_tmp_clean].add(line)

NUM = args.num
emoji_pair = []
for emoji_one in tqdm(emoji_thre_list):
    data = list(emoji_data[emoji_one])
    if len(data) < args.thre:
        continue

    for tmp in range(NUM):
        data_tmp = random.sample(data, 2)
        # emoji_pair.append(  {'text1':lines[data_tmp[0]].replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip(), \
        #                     'text2':lines[data_tmp[1]].replace('[RT] ', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()}  )
        emoji_pair.append({'text1': data_tmp[0],'text2': data_tmp[1]})

write_json('emoji_pair_thre'+str(args.thre)+'_num'+str(args.num), emoji_pair)