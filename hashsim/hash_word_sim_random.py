import json
from tqdm import tqdm
import random
with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)
hashtags = list(CONVERT.values())
TOP = 5
file = open('hash_word_sim_random.txt', 'a', encoding='utf-8')
for hashone in tqdm(hashtags):
    val = random.sample(hashtags, TOP)
    tmp_merge = '' + hashone
    for hash_tmp in val:
        tmp_merge = tmp_merge + '\t' + hash_tmp
    tmp_merge += ' \n'
    file.write(tmp_merge)
