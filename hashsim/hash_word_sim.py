import json
with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)
hashtags = CONVERT.values()

file = open('hash_word_sim.txt', 'a', encoding='utf-8')
for hashone in hashtags:
    hash_save = ''+hashone
    for hashtwo in hashtags:
        if hashone in hashtwo or hashtwo in hashone:
            hash_save += '\t'+hashtwo
    hash_save+=' \n'
    file.write(hash_save)