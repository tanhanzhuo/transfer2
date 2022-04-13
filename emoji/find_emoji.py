import emoji
from tqdm import tqdm
emoji_dic = {}
with open('/work/data/twitter_ref.txt', 'r') as f:
    for line in tqdm(f):
        emoji_list = emoji.distinct_emoji_lis(line)
        for emoji_one in emoji_list:
            if emoji_one in emoji_dic.keys():
                emoji_dic[emoji_one] += 1
            else:
                emoji_dic[emoji_one] = 1
emoji_sort = dict(sorted(emoji_dic.items(), key=lambda x: x[1]))
print(emoji_sort)
