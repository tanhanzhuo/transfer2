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
# from accelerate import Accelerator
# accelerate = Accelerator()
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='feature_modelT100N100_fileT100_num10',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory
args = parser.parse_args()

f=open('./selected_thre100_num500_index.json','r',encoding='utf-8')
hash_dic = json.load(f)
f.close()

def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line_sp = line.split(',')[0]
        hash_num = line_sp.split(':')[-1]
        hash_one = hash_dic[hash_num]
        data.append(hash_one)
    return data
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    center_samples=tmp['center_samples']
    # hash_embs.extend(tmp['center_embs'])
    center_embs=tmp['center_embs']
    tmp.close()
    hash_tags = read_data(args.hash_file+'_'+str(idx)+'.txt')
    print('samples:{},hashtags:{}'.format(len(center_samples),len(hash_tags)))
    # np.savez(args.hash_file + '_' + str(idx), center_samples=np.array(center_samples),
    #          center_embs=np.array(center_embs),hash_tags=np.array(hash_tags))
