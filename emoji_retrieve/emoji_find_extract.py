import re
import emoji
import json
import random
from tqdm import tqdm
TASK = 'sem19-task6-offen'
emoji_dic = {}
emoji_num = {}
################read the raw text
with open('./retrieve_'+TASK+'.json', 'r', encoding='utf-8') as f:
    for one in f:
        line = json.loads(one)['text']
        emoji_list = emoji.distinct_emoji_list(line)
        line_clean = emoji.replace_emoji(line)
        # for emoji_one in emoji_list:
        #     if emoji_one in emoji_dic.keys():
        #         emoji_dic[emoji_one] += 1
        #     else:
        #         emoji_dic[emoji_one] = 1
        if len(emoji_list) < 1:
            # print(line)
            continue
        emoji_one = emoji_list[0]
        if emoji_one in emoji_dic.keys():
            emoji_dic[emoji_one].append(line_clean)
            emoji_num[emoji_one] += 1
        else:
            emoji_dic[emoji_one] = [line_clean]
            emoji_num[emoji_one] = 1

emoji_num_sort = dict(sorted(emoji_num.items(), key=lambda x: x[1],reverse=True))
emoji_top = list(emoji_num_sort.keys())[:20]

label2idx = {}
for idx in range(len(emoji_top)):
    label2idx[emoji_top[idx]] = idx

USER = re.compile(r"@\S+")

data_emoji = []
for emoji_one in emoji_top:
    data_raw = emoji_dic[emoji_one]
    for data_one in data_raw:
        user_tmp = USER.findall(data_one)
        data_tmp = data_one[:]
        for user in user_tmp:
            data_tmp = data_tmp.replace(user+' ','')
        if len(data_tmp.strip().split(' ')) > 5:
            data_emoji.append({'label': label2idx[emoji_one], 'text': data_one})

random.shuffle(data_emoji)
SP = int(len(data_emoji)/10)
with open('train_'+TASK+'.json', 'a', encoding='utf-8') as f:
    for tweet in data_emoji[:9*SP]:
        json.dump(tweet, f)
        f.write('\n')
with open('dev_'+TASK+'.json', 'a', encoding='utf-8') as f:
    for tweet in data_emoji[9*SP:]:
        json.dump(tweet, f)
        f.write('\n')
######################format the data
# with open('data_emoji.txt', 'w') as f_write:
#     with open('/work/data/twitter_ref.txt', 'r') as f_read:
#         for line in tqdm(f_read):
#             emoji_list = emoji.distinct_emoji_list(line)
#             if len(emoji_list) == 1:
#                 emoji_one = emoji_list[0]
#                 if emoji_one in emoji_dic.keys():
#                     emoji_dic[emoji_one] += 1
#                 else:
#                     emoji_dic[emoji_one] = 1
#                 line_clean = emoji.replace_emoji(line)
#                 line_clean = line_clean.replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]', 'https')
#                 f_write.write(emoji_one + '\t' + line_clean)
#
# emoji_dic_sort = dict(sorted(emoji_dic.items(), key=lambda x: x[1]))
# print(emoji_dic_sort)
#
#
# # emoji_top = {'👀':0, '🙌':0, '😳':0, '💀':0, '😔':0, '😏':0, '😁':0, '❤':0, '🔥':0, '👌':0,
# #         '😒':0, '💯':0, '💕':0, '😘':0, '😊':0, '😩':0, '❤️':0, '😍':0, '😭':0, '😂':0}
# emoji_top = ['🤔', '🙄', '😳', '👌', '😁', '😏', '💯', '🔥', '💕', '😘', '😔', '❤', '♥', '😒', '😊', '😩', '❤️', '😍', '😭', '😂']
# emoji_top_cluster = [['😁', '😏', '💕', '😘', '😊', '😍'], ['👌', '💯', '🔥'], ['😳', '😔', '😒', '😩'], ['❤', '♥', '❤️'], ['🤔', '🙄'], ['😭', '😂']]
#
# label2idx = {}
# for idx in range(len(emoji_top)):
#     label2idx[emoji_top[idx]] = idx
# with open('data_emoji.txt', 'r') as f:
#     data_emoji = f.readlines()
#
# emoji_num = [0] * len(emoji_top_cluster)
# data_emoji_top = []
# for data_one in tqdm(data_emoji):
#     emoji_one = data_one.split('\t')[0]
#     line = data_one.split('\t')[1].strip().replace('  ',' ')
#     if emoji_one in emoji_top:
#         if len(line.split(' ')) > 5:
#             txt = line.replace('https://', 'https') + '\n'
#             lab = -1
#             for idx_lab in range(len(emoji_top_cluster)):
#                 if emoji_one in emoji_top_cluster[idx_lab]:
#                     lab = idx_lab
#                     emoji_num[lab]+=1
#             if lab == -1:
#                 print('****************ERROR***************')
#             data_emoji_top.append(
#                 {'label': lab, 'text': txt}
#             )
# print(emoji_num)
# random.shuffle(data_emoji_top)
# SP = int(len(data_emoji_top)*0.9)
# with open('./cluster_train/train.json', 'w') as f:
#     for idx in trange(SP):
#         json.dump(data_emoji_top[idx], f)
#         f.write('\n')
# with open('./cluster_train/dev.json', 'w') as f:
#     for idx in trange(SP,len(data_emoji_top)):
#         json.dump(data_emoji_top[idx], f)
#         f.write('\n')