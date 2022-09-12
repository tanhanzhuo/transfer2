import json
from tqdm import tqdm
with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)
hashtags = CONVERT.values()

file = open('hash_word_sim.txt', 'a', encoding='utf-8')
for hashone in tqdm(hashtags):
    hash_save = ''+hashone
    for hashtwo in hashtags:
        if hashone in hashtwo:
            hash_save += '\t'+hashtwo
        elif hashtwo in hashone and len(hashtwo)>2:
            hash_save += '\t' + hashtwo
    hash_save+=' \n'
    file.write(hash_save)