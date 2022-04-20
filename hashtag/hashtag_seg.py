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
    # reranker_model_name_or_path="bert-base-uncased"
)

for one in tqdm(data):
    segmentations = ws.segment([one])
    print(segmentations)

BS = 16
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    segmenter_gpu_batch_size=BS
    # reranker_model_name_or_path="bert-base-uncased"
)

data_seg = []
NUM = int(len(data)/BS)
for idx in range(NUM):
    data_seg.append( data[idx*BS:(idx+1)*BS] )
data_seg.append( data[NUM*BS:] )
for one in tqdm(data_seg):
    segmentations = ws.segment(one)
    print(segmentations)