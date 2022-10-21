# def readhash(fileName):
#     with open(fileName, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         hashtags = {}
#         for line in lines:
#             tmp = line.strip().split('\t')
#             hashtags[tmp[0]] = tmp[1:]
#             if len(tmp) != 6:
#                 print(line)
#     return hashtags
# hash_select = readhash('hash_select.txt')
# hash_bt = readhash('hash_merge_seg_cluster.txt')
# hash_simcse = readhash('hash_merge_simcse_seg_cluster.txt')
# hash_glove = readhash('hash_word_sim_glove.txt')
# hash_random = readhash('hash_word_sim_random.txt')
# hash_relation = readhash('hash_word_relation.txt')
# hash_wordbt = readhash('hash_word_sim_bt.txt')
#
# hash_all = {}
# hash_all_set = set()
# for hash_one in hash_select.keys():
#     hash_all[hash_one] = hash_select[hash_one][:3] + hash_simcse[hash_one][:3] + hash_glove[hash_one][:3]\
#                         +hash_random[hash_one][:3] + hash_relation[hash_one][:3] + hash_wordbt[hash_one][:3]
#     hash_all[hash_one] = list(set(hash_all[hash_one]))
#     hash_all_set.add(hash_one)
#     for tmp_one in hash_all[hash_one]:
#         hash_all_set.add(tmp_one)
# hash_all_set = list(hash_all_set)
#
# import json
# from tqdm import tqdm,trange
# with open('../contrastive_full/thre100_index.json', 'r', encoding='utf-8') as f:
#     CONVERT = json.load(f)

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


###################### read and save hash samples for cluster


# import numpy as np
# hash_samples = []
# hash_tags = []
# for idx in trange(4):
#     tmp = np.load('feature_modelT100N100M_fileT100N100S_num10_cluster_'+str(idx)+'.npz',allow_pickle=True)
#     hash_samples.extend(tmp['center_samples'])
#     hash_tags.extend(tmp['center_hash'])
#     tmp.close()
#
# hash_samples = np.reshape(hash_samples, [len(hash_tags),10])
# hash_sample = {}
# for hash_idx in trange(len(hash_tags)):
#     hash_one = hash_tags[hash_idx]
#     if hash_one not in hash_all_set:
#         continue
#     hash_sample[hash_one] = list(hash_samples[hash_idx])
#
# with open('hash_sample_bt_cluster10.json', 'w', encoding='utf-8') as f:
#     json.dump(hash_sample, f)


###################### read and save hash samples for cluster


# import numpy as np
# hash_samples = []
# hash_tags = []
# for idx in trange(4):
#     tmp = np.load('feature_simcse_fileT100N100S_num10_cluster_'+str(idx)+'.npz',allow_pickle=True)
#     hash_samples.extend(tmp['center_samples'])
#     hash_tags.extend(tmp['center_hash'])
#     tmp.close()
#
# hash_samples = np.reshape(hash_samples, [len(hash_tags),10])
# hash_sample = {}
# for hash_idx in trange(len(hash_tags)):
#     hash_one = hash_tags[hash_idx]
#     if hash_one not in hash_all_set:
#         continue
#     hash_sample[hash_one] = list(hash_samples[hash_idx])
#
# with open('hash_sample_simcse_cluster10.json', 'w', encoding='utf-8') as f:
#     json.dump(hash_sample, f)



########################## select the final samples

# with open('hash_sample.json', 'r', encoding='utf-8') as f:
#     hash_sample = json.load(f)
#
# with open('hash_sample_bt_cluster10.json', 'r', encoding='utf-8') as f:
#     hash_sample_bt = json.load(f)
#
# with open('hash_sample_simcse_cluster10.json', 'r', encoding='utf-8') as f:
#     hash_sample_simcse = json.load(f)
#
# NUM1 = 10
# NUM2 = 10
# import random
# hash_sample_merge = {}
# for hash_one in list(hash_sample.keys()):
#     hash_sample[hash_one] = list(set(hash_sample[hash_one]))
#     hash_sample_merge[hash_one] = set()
#     hash_sample_random = random.sample(hash_sample[hash_one],NUM2)
#
#     for i in random.sample(range(NUM2),NUM1):
#         hash_sample_merge[hash_one].add(hash_sample_random[i])
#         # hash_sample_merge[hash_one].add(hash_sample_bt[hash_one][i])
#         # hash_sample_merge[hash_one].add(hash_sample_simcse[hash_one][i])
#     # hash_sample_merge[hash_one] = random.sample( list(hash_sample_merge[hash_one]), NUM2 )
#     hash_sample_merge[hash_one] =list(hash_sample_merge[hash_one])
#     if len(hash_sample_merge[hash_one]) < NUM2:
#         print('!!!!!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!!!!!')
#
# with open('hash_sample_random10.json', 'w', encoding='utf-8') as f:
#     json.dump(hash_sample_merge, f)
#
# with open('hash_select_all.json', 'w', encoding='utf-8') as f:
#     json.dump(hash_all, f)
#
# hash_all_keys = list(hash_all.keys())
# random.shuffle(hash_all_keys)
# with open('hash_pair.txt', 'w', encoding='utf-8') as f:
#     for hash_one in hash_all_keys:
#         for hash_two in hash_all[hash_one]:
#             f.write(hash_one + '\t' + hash_two + '\n')


############################################ write comments to the hashtag

with open('hash_pair.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip().split('\t')
        for one in line:
            if one not in data:
                data.append(one)

with open('hash_comment.txt', 'w', encoding='utf-8') as f:
    for hash_one in data:
        f.write(hash_one + '\t' + hash_one + '\n')


######################### display two hashtags with samples
# import json
#
# with open('hash_select_all.json', 'r', encoding='utf-8') as f:
#     hash_all = json.load(f)
#
# with open('hash_pair.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     data = []
#     for line in lines:
#         line = line.strip().split('\t')
#         data.append(line)
#
# with open('hash_sample_random10.json', 'r', encoding='utf-8') as f:
#     hash_sample_random = json.load(f)
#
# from prettytable import PrettyTable
#
# NUM=10
# for idx in range(len(data)):
#     pair = data[idx]
#     if len(pair) == 3:
#         continue
#     if len(pair) != 2:
#         print('!!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!')
#     hash_one = pair[0]
#     hash_two = pair[1]
#     table = PrettyTable(['IDX',hash_one,hash_two],max_width=50)
#     table.hrules=True
#     table.align = 'l'
#     for idx in range(NUM):
#         table.add_row([str(idx), hash_sample_random[hash_one][idx], hash_sample_random[hash_two][idx]])
#     print(table)
#     print('In *% of tweets, these two hashtag can be replaced by each other, please select from [1,3,5,7,9]')
#     print('1: around 10%; 3: around 30%; 5: around 50%; 7: around 70%; 9: around 90%')
#     rate = input('type your rate: ')
#     pair.append(rate)
#
#     with open('hash_pair.txt', 'w', encoding='utf-8') as f:
#         for pair in data:
#             f.write('\t'.join(pair) + '\n')

####### write down all the hash pairs

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