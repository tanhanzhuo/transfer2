import emoji
from tqdm import tqdm
emoji_dic = {}
emoji_num = {}
# with open('/work/data/twitter_ref.txt', 'r') as f:
#     for line in tqdm(f):
#         emoji_list = emoji.distinct_emoji_list(line)
#         for emoji_one in emoji_list:
#             if emoji_one in emoji_dic.keys():
#                 emoji_dic[emoji_one] += 1
#             else:
#                 emoji_dic[emoji_one] = 1
#         if len(emoji_list) in emoji_num.keys():
#             emoji_num[len(emoji_list)] += 1
#         else:
#             emoji_num[len(emoji_list)] = 1
# emoji_dic_sort = dict(sorted(emoji_dic.items(), key=lambda x: x[1]))
# emoji_num_sort = dict(sorted(emoji_num.items(), key=lambda x: x[0]))
# print(emoji_dic_sort)
# print(emoji_num_sort)
with open('data_emoji.txt', 'w') as f_write:
    with open('/work/data/twitter_ref.txt', 'r') as f_read:
        for line in tqdm(f_read):
            emoji_list = emoji.distinct_emoji_list(line)
            if len(emoji_list) == 1:
                emoji_one = emoji_list[0]
                if emoji_one in emoji_dic.keys():
                    emoji_dic[emoji_one] += 1
                else:
                    emoji_dic[emoji_one] = 1
                line_clean = emoji.replace_emoji(line)
                line_clean = line_clean.replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]', 'https')
                f_write.write(emoji_one + '\t' + line_clean)

emoji_dic_sort = dict(sorted(emoji_dic.items(), key=lambda x: x[1]))
print(emoji_dic_sort)
