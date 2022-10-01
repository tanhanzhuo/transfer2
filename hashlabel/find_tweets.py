def readhash(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        hashtags = {}
        for line in lines:
            tmp = line.strip().split('\t')
            hashtags[tmp[0]] = tmp[1:]
            if len(tmp) != 6:
                print(line)
    return hashtags
hash_select = readhash('hash_select.txt')
hash_bt = readhash('hash_merge_seg_cluster.txt')
hash_simcse = readhash('hash_merge_simcse_seg_cluster.txt')
hash_glove = readhash('hash_word_sim_glove.txt')

hash_all = {}
hash_all_set = set()
for hash_one in hash_select.keys():
    hash_all[hash_one] = hash_select[hash_one][:3] + hash_simcse[hash_one][:3] + hash_glove[hash_one][:3]
    hash_all_set.add(hash_one)
    for tmp_one in hash_all[hash_one]:
        hash_all_set.add(tmp_one)
hash_all_set = list(hash_all_set)

import json
from tqdm import tqdm,trange
with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)

######################## read and save hash samples
# hash_sample = {}
# for SP in range(4):
#     with open('twitter_hash_sep_thre100_num100_'+str(SP)+'.json', 'r', encoding='utf-8') as f:
#
#         for line in tqdm(f):
#             hash_tmp = json.loads(line)
#             if CONVERT[str(hash_tmp['labels'])] not in hash_all_set:
#                 continue
#             tmp = hash_sample.get(CONVERT[str(hash_tmp['labels'])])
#             if tmp is not None:
#                 hash_sample[CONVERT[str(hash_tmp['labels'])]].append(hash_tmp['text'])
#             else:
#                 hash_sample[CONVERT[str(hash_tmp['labels'])]] = [hash_tmp['text']]
#
# with open('hash_sample.json', 'w', encoding='utf-8') as f:
#     json.dump(hash_sample, f)

with open('hash_sample.json', 'r', encoding='utf-8') as f:
    hash_sample = json.load(f)

#
import random
for hash_one in list(hash_all.keys()):
    for hash_two in hash_all[hash_one]:
        print('hashtag one:{}'.format(hash_one))
        for i in random.sample(hash_sample[hash_one],10):
            print(i)

        print('hashtag two:{}'.format(hash_two))
        for i in random.sample(hash_sample[hash_two], 10):
            print(i)
        print('Please type the similarity:')

######## write down all the hash pairs
# with open('hash_select_all.txt', 'w', encoding='utf-8') as f:
#     for hash_one in list(hash_all.keys()):
#         tmp = hash_one+'\t'+'\t'.join(hash_all[hash_one])
#         f.write(tmp+'\n')

######## find hashtag with several words
# with open('../contrastive/hash_seg.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
# hash_seg = {}
# for line in lines:
#     if len(line.split('\t')[1].split(' ')) > 1:
#         hash_seg[line.split('\t')[0]] = line.split('\t')[1]
#
# hash_select_twowords = {}
# for hash_one in hash_seg.keys():
#     tmp = hash_bt.get(hash_one)
#     if tmp is not None:
#         hash_select_twowords[hash_one] = hash_bt[hash_one][:3] + hash_simcse[hash_one][:3] + hash_glove[hash_one][:3]
# print(0)