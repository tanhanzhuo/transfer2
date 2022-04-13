import emoji
from tqdm import tqdm
emoji_dic = {}
emoji_num = {}
with open('/work/data/twitter_ref.txt', 'r') as f:
    for line in tqdm(f):
        emoji_list = emoji.distinct_emoji_list(line)
        for emoji_one in emoji_list:
            if emoji_one in emoji_dic.keys():
                emoji_dic[emoji_one] += 1
            else:
                emoji_dic[emoji_one] = 1
        if len(emoji_list) in emoji_num.keys():
            emoji_num[emoji_list] += 1
        else:
            emoji_num[emoji_list] = 1
emoji_dic_sort = dict(sorted(emoji_dic.items(), key=lambda x: x[1]))
emoji_num_sort = dict(sorted(emoji_num.items(), key=lambda x: x[0]))
print(emoji_dic_sort)
print(emoji_num_sort)
