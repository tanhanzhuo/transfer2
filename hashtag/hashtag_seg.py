import os
import string
import torch
from tqdm import tqdm, trange
import re
data ={}
num = []
with open('hash_his.txt', 'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        hash = line.split('\t')[0].lower()
        if hash[-1] == '…':
            continue
        if len(hash) > 3 and  hash[-3:] == '...':
            continue
        if hash[-1] in string.punctuation:
            hash = hash[:-1]
        hash_clean = re.findall('[a-z0-9]*', hash)
        hash_clean = '#'+''.join(hash_clean)
        if hash == hash_clean:
            if hash in data.keys():
                data[hash] += int(line.split('\t')[1])
            else:
                data[hash] = int(line.split('\t')[1])


data_hash = list(data.keys())

from hashformers import TransformerWordSegmenter as WordSegmenter
with torch.no_grad():
    data_seg = []
    ws = WordSegmenter(
        segmenter_model_name_or_path="distilgpt2"
        # reranker_model_name_or_path="bert-base-uncased"
    )

    with open('hash_seg.txt', 'a') as f:
        for idx in trange(int(len(data_hash)/2)):
            segmentations = ws.segment([data_hash[idx]])
            f.write(data_hash[idx] + '\t' + segmentations[0] + '\t' + data[data_hash[idx]] + '\n')
            if (idx+1) % 10000 ==0:
                del ws,segmentations
                torch.cuda.empty_cache()
                ws = WordSegmenter(segmenter_model_name_or_path="distilgpt2")