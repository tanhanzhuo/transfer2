import os
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
    for idx in range(len(SELECT)):
        label2idx[SELECT[idx]] = idx



    with open('data_hash_' + str(SELECT_HASH) + '_' + str(SELECT_NUM) + '.txt', 'r') as f:
        data_hash = f.readlines()
    data_hash_top = []
    for data_one in data_hash:
        hash_one = data_one.split('\t')[0]
        line = data_one.split('\t')[1].strip().replace('  ',' ')
        line = line.replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]', 'https')
        if len(line.split(' ')) > 5:
            txt = line+ '\n'
            lab = label2idx[hash_one]
            data_hash_top.append(
                {'label': lab, 'text': txt}
            )
    
    random.shuffle(data_hash_top)
    SP = int(len(data_hash_top)*0.9)
    os.makedirs('data_hash_' + str(SELECT_HASH) + '_' + str(SELECT_NUM), exist_ok=True)
    with open('./data_hash_' + str(SELECT_HASH) + '_' + str(SELECT_NUM) + '/train.json', 'w') as f:
        for idx in range(SP):
            json.dump(data_hash_top[idx], f)
            f.write('\n')
    with open('./data_hash_' + str(SELECT_HASH) + '_' + str(SELECT_NUM) + '/dev.json', 'w') as f:
        for idx in range(SP,len(data_hash_top)):
            json.dump(data_hash_top[idx], f)
            f.write('\n')