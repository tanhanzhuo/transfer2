import json
from tqdm import tqdm
with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)
hashtags = list(CONVERT.values())

with open('../contrastive/hash_seg.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

hashtags_seg = []
for hashone in hashtags:
    tmp = hash_seg.get(hashone)
    if tmp is None:
        hashtags_seg.append(hashone[1:])
    else:
        hashtags_seg.append(tmp)


import numpy as np
import torch
from tqdm import tqdm,trange
word_embeddings = {}
f = open('glove.6B.300d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

hash_vectors = []
for i in hashtags_seg:
    if len(i) != 0:
        v = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split()))
    else:
        v = np.zeros((300,))
    hash_vectors.append(v)

hash_vectors = torch.tensor(np.array(hash_vectors)).cuda()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
TOP=5
file = open('hash_word_sim_glove.txt', 'a', encoding='utf-8')
with torch.no_grad():
    for idx in trange(10000):
        dis = cos_sim(hash_vectors[idx],hash_vectors)
        dis[idx] = 0
        val, place = dis.topk(TOP)
        tmp_merge = '' + hashtags[idx]
        for idx_tmp in range(TOP):
            tmp_merge = tmp_merge + '\t' + hashtags[place[idx_tmp].item()]
        tmp_merge += ' \n'
        file.write(tmp_merge)
file.close()