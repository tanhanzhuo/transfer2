import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import os
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--tasks',default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance',type=str)
parser.add_argument('--sp',default='_clean_111_',type=str)
parser.add_argument('--method',default='_fulldata_simcse_top20_textfirst',type=str)#'_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst'
args = parser.parse_args()

for task in args.tasks.split(','):
    print(task)
    data_source = []
    data_source_text = []
    for sp in ['train', 'dev', 'test']:
        with open('../finetune/data/' + task + '/' + sp + args.method + '.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                one = json.loads(line)
                data_source.append(one)
                data_source_text.append(one['text'].split(' \n ')[-2].strip())

    for epoch in trange(0,10):

        for sp in ['train', 'dev', 'test']:
            data_sem = []
            with open('../finetune/data/' + task+args.sp+str(epoch) + '/' + sp + '.json', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    one = json.loads(line)
                    data_sem.append(one['text'].strip())

            with open('../finetune/data/' + task + args.sp + str(epoch) + '/' + sp + args.method + '.json', 'w',
                      encoding='utf-8') as f:
                for idx in range(len(data_sem)):
                    one_text = data_sem[idx]
                    match = 0
                    for idx2 in range(len(data_source_text)):
                        two_text = data_source_text[idx2]
                        if one_text == two_text:
                            tmp = json.dumps(data_source[idx2], ensure_ascii=False)
                            f.write(tmp + '\n')
                            match = 1
                            break
                    if match == 0:
                        print(one_text)

