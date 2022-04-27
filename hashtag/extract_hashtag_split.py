import random
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hash', default=100,type=int)
parser.add_argument('--num', default=1000,type=int)

if __name__ == "__main__":

    args = parser.parse_args()
    SELECT_HASH = args.hash
    SELECT_NUM = args.num

    with open('hash_his.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    hash_dic = {}
    for line in lines:
        line = line.strip()
        hashtag = line.split('\t')[0]
        num = int(line.split('\t')[1])
        if num > 10000:
            hash_dic[hashtag] = num
    hash_dic_sort = dict(sorted(hash_dic.items(), key=lambda x: x[1], reverse=True))
    SELECT = list(hash_dic_sort.keys())[:SELECT_HASH]



label2idx = {}
for idx in range(len(emoji_top)):
    label2idx[emoji_top[idx]] = idx
with open('data_emoji.txt', 'r') as f:
    data_emoji = f.readlines()

data_emoji_top = []
for data_one in data_emoji:
    emoji_one = data_one.split('\t')[0]
    line = data_one.split('\t')[1].strip().replace('  ',' ')
    if emoji_one in emoji_top:
        if len(line.split(' ')) > 5:
            txt = line.replace('https://', 'https') + '\n'
            lab = label2idx[emoji_one]
            data_emoji_top.append(
                {'label': lab, 'text': txt}
            )

random.shuffle(data_emoji_top)
SP = int(len(data_emoji_top)*0.9)
with open('train.json', 'w') as f:
    for idx in range(SP):
        json.dump(data_emoji_top[idx], f)
        f.write('\n')
with open('dev.json', 'w') as f:
    for idx in range(SP,len(data_emoji_top)):
        json.dump(data_emoji_top[idx], f)
        f.write('\n')