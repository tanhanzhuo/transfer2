import os
import string
import torch
from tqdm import tqdm, trange
import json
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='tweet_hash_clean_group.txt',type=str)
parser.add_argument('--num',default=20,type=int)

args = parser.parse_args()

data =[]
num = []
f=open('./hash_his.json','r',encoding='utf-8')
hash_dic = json.load(f)
f.close()
for hash_one in list(hash_dic.keys()):
    if hash_dic[hash_one] < args.num:
        hash_dic.pop(hash_one)

from hashformers import TransformerWordSegmenter as WordSegmenter
with torch.no_grad():
    data_seg = []
    ws = WordSegmenter( segmenter_model_name_or_path="gpt2",reranker_model_name_or_path="bert-base-uncased")

    with open('hash_seg'+str(args.num)+'.txt', 'a') as f:
        for hash_one in tqdm(hash_dic.keys()):
            segmentations = ws.segment([hash_one])
            f.write(hash_one + '\t' + segmentations[0] + '\t' + str(hash_dic[hash_one]) + '\n')