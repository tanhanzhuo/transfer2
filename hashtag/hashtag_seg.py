import os
from tqdm import tqdm, trange
import json
data =[]
with open('hash_his.json', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(json.loads(line))
from hashformers import TransformerWordSegmenter as WordSegmenter
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path="bert-base-uncased"
)
segmentations = ws.segment([
    "#weneedanationalpark",
    "#icecold"
])
