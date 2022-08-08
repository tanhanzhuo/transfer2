import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import time
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='./hash_convert.json',type=str)
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=1, type=int)
parser.add_argument('--method',default='_modelT100N100S_fileT100S_num10_cluster',type=str)
#simcse
args = parser.parse_args()

import json
def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data
def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

f=open(args.hash_file,'r',encoding='utf-8')
hash_convert = json.load(f)
f.close()

for task in tqdm(args.task_name.split(',')):
    for fileName in ['train', 'dev', 'test']:
        for top in range(args.best):
            dataset = read_data(args.dataset_path + task + '/' + fileName + args.method + '_top' + str(top) \
                           + '_' + 'hashlast.json')
            dataset_up = copy.deepcopy(dataset)
            for idx_one in range(len(dataset)):
                data_one = dataset[idx_one]
                text_sp = data_one['text'].strip().split(' ')
                hash = text_sp[-1]
                tmp = hash_convert.get(hash)
                if tmp is None:
                    print(hash)
                    continue
                text = ' '.join(text_sp[:-1]) + ' ' + hash_convert[hash]
                dataset_up[idx_one]['text'] = text
            write_json(dataset_up, args.dataset_path + task + '/' + fileName + args.method + '_top' + str(top) \
                           + '_' + 'hashlast_up')

            dataset = read_data(args.dataset_path + task + '/' + fileName + args.method + '_top' + str(top) \
                                + '_' + 'hashfirst.json')
            dataset_up = copy.deepcopy(dataset)
            for idx_one in range(len(dataset)):
                data_one = dataset[idx_one]
                text_sp = data_one['text'].strip().split(' ')
                hash = text_sp[0]
                tmp = hash_convert.get(hash)
                if tmp is None:
                    print(hash)
                    continue
                text = hash_convert[hash]+ ' '+' '.join(text_sp[1:])
                dataset_up[idx_one]['text'] = text
            write_json(dataset_up, args.dataset_path + task + '/' + fileName + args.method + '_top' + str(top) \
                       + '_' + 'hashfirst_up')
