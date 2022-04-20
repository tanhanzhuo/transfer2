import os
from tqdm import tqdm, trange
data =[]
num = []
with open('hash_his.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        data.append(line.split('\t')[0])
        num.append(line.split('\t')[1])

from hashformers import TransformerWordSegmenter as WordSegmenter
data_seg = []
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    # reranker_model_name_or_path="bert-base-uncased"
)

with open('hash_seg.txt', 'a') as f:
    for idx in trange(int(len(data)/2)):
        segmentations = ws.segment([data[idx]])
        f.write(data[idx] + '\t' + segmentations[0] + '\t' + num[idx] + '\n')
