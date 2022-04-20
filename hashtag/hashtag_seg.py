import os
from tqdm import tqdm, trange
data =[]
with open('hash_his.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line.split('\t')[0])

from hashformers import TransformerWordSegmenter as WordSegmenter
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path="bert-base-uncased"
).cuda()

for one in tqdm(data):
    segmentations = ws.segment([
        one
    ])
