import argparse
from scipy.spatial.distance import pdist, squareform,cosine
import numpy as np
import torch
from tqdm import tqdm,trange
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='../contrastive_full/feature_modelT100N100M_fileT100N100S_num10',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory
parser.add_argument("--hash_pair", default='hash_pair.txt', type=str)
parser.add_argument("--num", default=500, type=int)
args = parser.parse_args()

fileName = args.hash_pair
with open(fileName, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = {}
    for line in lines:
        line = line.strip().split('\t')
        if len(line) != 3:
            continue
        if line[0] in data.keys():
            data[line[0]][line[1]] = float(line[2]) * 0.25 +0.125
        else:
            data[line[0]] = {line[1]:float(line[2]) * 0.25 +0.125}


import copy
data_hash = copy.deepcopy(data)
with torch.no_grad():
    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    hash_embs = []
    hash_tags = []
    for idx in range(args.split):
        tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
        # hash_embs.extend(tmp['center_embs'])
        hash_embs.extend(tmp['center_embs'])
        hash_tags.extend(tmp['center_hash'])
        tmp.close()
    hash_embs = torch.tensor(np.array(hash_embs)).cuda()
    [num,dim]=hash_embs.shape
    hash_embs = hash_embs.reshape(int(num/10),10,dim)

    for hash_one in data.keys():
        id1 = hash_tags.index(hash_one)
        emb1 = hash_embs[id1]
        for hash_two in data[hash_one].keys():
            id2 = hash_tags.index(hash_two)
            emb2 = hash_embs[id2:id2+1]
            dis_tmp = cos_sim(emb1, emb2.unsqueeze(2)).mean(dim=[1, 2])
            data_hash[hash_one][hash_two] = dis_tmp[0].item()

data_score = []
data_hash_score = []
for hash_one in data.keys():
    for hash_two in data[hash_one].keys():
        data_score.append(data[hash_one][hash_two])
        data_hash_score.append(data_hash[hash_one][hash_two])

idx = np.array(range(len(data_score)))
np.random.shuffle(idx)
data_score = np.array(data_score)
data_hash_score = np.array(data_hash_score)
data_score = data_score[idx]
data_hash_score = data_hash_score[idx]

import scipy.stats
for num in [100,200,300,400,500]:
    print(scipy.stats.pearsonr(data_score[:num], data_hash_score[:num])[0])  # Pearson's r
    print(scipy.stats.spearmanr(data_score[:num], data_hash_score[:num])[0])   # Spearman's rho
    print(scipy.stats.kendalltau(data_score[:num], data_hash_score[:num])[0])  # Kendall's tau