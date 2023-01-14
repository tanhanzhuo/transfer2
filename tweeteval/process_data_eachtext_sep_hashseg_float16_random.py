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
# from accelerate import Accelerator
# accelerate = Accelerator()
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='feature_modelT100N100R_fileT100N100R_num10',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
parser.add_argument('--model',default='/work/SimCSE-main/result/thre100_num100_remove/1399999',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=20, type=int)
parser.add_argument("--num_samples", default=10, type=int)
parser.add_argument("--word", default=False, type=bool)
parser.add_argument('--method',default='_modelT100N100R_fileT100N100R_num10',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory
#simcse
parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

args = parser.parse_args()

with open('../contrastive/hash_seg.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

# f=open('./selected_thre100_num500_index.json','r',encoding='utf-8')
# hash_dic = json.load(f)
# f.close()
# f=open('./selected_thre100_num3000_word_nltk.json','r',encoding='utf-8')
# hash_word = json.load(f)
# f.close()

def read_data(fileName):
    with open(fileName+'.json', 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data
import re
HASH = re.compile(r"#\S+")
def read_data_hashseg(fileName):
    with open(fileName+'.json', 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            one_dic = json.loads(line)
            one = one_dic['text']
            hash_tmp = HASH.findall(one)
            # for hash_two in hash_tmp:
            #     one = one.replace(hash_two, '')
            for hash_two in hash_tmp:
                tmp2 = hash_seg.get(hash_two.lower())
                if tmp2 is not None:
                    one = one.replace(hash_two, tmp2)
                else:
                    one = one.replace(hash_two, hash_two[1:])
            data.append({'labels': one_dic['labels'], 'text':one })
    return data
import json
def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

hash_samples = []
hash_embs = []
hash_tags = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.extend(tmp['center_samples'])
    # hash_embs.extend(tmp['center_embs'])
    # hash_embs.append(torch.tensor(tmp['center_embs']))
    tmp.close()

for task in args.task_name.split(','):
    for fileName in ['train', 'dev', 'test']:
    # for fileName in ['test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName)
        data_hash_all = copy.deepcopy(train_dataset)
        train_dataset = read_data_hashseg(args.dataset_path + task + '/' + fileName) ###remove hash to retrieve
        for idx in trange(len(train_dataset)):
            one = train_dataset[idx]
        best_text = random.sample(hash_samples, args.best)
        for cur_idx in range(args.best):
            data_hash_all[idx]['text'] = ' ' + best_text[cur_idx].strip() \
                                         + ' \n ' + data_hash_all[idx]['text'].strip() + ' \n '
        write_json(data_hash_all, args.dataset_path + task + '/' + fileName + '_random_top' + str(args.best) \
               + '_' + 'textfirst')

    print('task done! {}'.format(task))