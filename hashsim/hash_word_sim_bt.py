import json
import random
import re
import torch
import numpy as np
from tqdm import trange,tqdm
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

from transformers import AutoModel,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
model = AutoModel.from_pretrained('vinai/bertweet-base').cuda()


with torch.no_grad():
    hash_vectors = torch.tensor([[]]).view(-1,768).cuda()
    for idx in trange(len(hashtags_seg)):
        word = hashtags_seg[idx]
        hash_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        hash_emb = model.embeddings.word_embeddings(torch.tensor(hash_token).cuda()).detach()
        hash_emb_mean = torch.mean(hash_emb, dim=0, keepdim=True)
        hash_vectors = torch.cat((hash_vectors,hash_emb_mean),0)

    # hash_vectors = hash_vectors.cuda()
    cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
    TOP=5
    file = open('hash_word_sim_bt.txt', 'a', encoding='utf-8')
    with torch.no_grad():
        for idx in trange(len(hash_vectors)):
            if hash_vectors[idx].sum().item() != 0:
                dis = cos_sim(hash_vectors[idx],hash_vectors)
                dis[idx] = 0
                val, place = dis.topk(TOP)
                tmp_merge = '' + hashtags[idx]
                for idx_tmp in range(TOP):
                    tmp_merge = tmp_merge + '\t' + hashtags[place[idx_tmp].item()]
                tmp_merge += ' \n'
                file.write(tmp_merge)
            else:
                val = random.sample(hashtags, TOP)
                tmp_merge = '' + hashtags[idx]
                for hash_tmp in val:
                    tmp_merge = tmp_merge + '\t' + hash_tmp
                tmp_merge += ' \n'
                file.write(tmp_merge)
    file.close()