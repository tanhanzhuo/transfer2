import json
import random

import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import time
import copy
import faiss

parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='./tweet_hash_clean_seg_500_one20/tweet_hash_clean_seg_500',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
# parser.add_argument('--model_name',default='/work/SimCSE-main/result/thre100_num100_remove/1399999',type=str)
parser.add_argument('--model_name',default='../lmbff/contrastive_models/one/20_new/',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=100, type=int)
parser.add_argument('--method',default='_random',type=str)
parser.add_argument("--split", default=50, type=int)#for gpu memory
parser.add_argument("--hashprocess", default='seg', type=str)#for gpu memory
parser.add_argument("--gpu", default=8, type=int)#for gpu memory

args = parser.parse_args()


def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            # data.append({'labels': line.split('\t')[0], 'text': line.split('\t')[1]})
            data.append(json.loads(line))
    return data

import json
def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

hash_samples = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.extend(tmp['samples'].tolist())
    tmp.close()

for task in args.task_name.split(','):
    for fileName in ['train', 'dev', 'test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName + '.json')
        data_hash_all = copy.deepcopy(train_dataset)
        for idx in range(len(train_dataset)):
            best_idx = random.sample(range(len(hash_samples)),args.best)
            for cur_idx in range(0,len(best_idx)):
                data_hash_all[idx]['text'+str(cur_idx)] = hash_samples[best_idx[cur_idx]].strip()

        write_json(data_hash_all, args.dataset_path + task + '/' + fileName + args.method + '_top' + str(args.best) + '_sp')

    print('task done! {}'.format(task))