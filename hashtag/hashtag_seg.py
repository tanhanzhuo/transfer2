import os
from tqdm import tqdm, trange
import re
from hashformers import TransformerWordSegmenter as WordSegmenter
ws = WordSegmenter(
    segmenter_model_name_or_path="gpt2",
    reranker_model_name_or_path="bert-base-uncased"
)
segmentations = ws.segment([
    "#weneedanationalpark",
    "#icecold"
])

HASH = re.compile(r"#\S+")
hash_dic = {}
filePath = '/work/test/twitter_hash.txt'
with open(filePath, 'r') as f:
    for line in f:
        if not line:
            continue
        hash_tmp = HASH.findall(line)
        for hash_one in hash_tmp:
            if hash_one[1].isalpha():
