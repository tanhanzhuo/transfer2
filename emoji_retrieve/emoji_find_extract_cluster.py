import re
import emoji
import json
import random
from tqdm import tqdm
import os
TASK = 'sem19-task6-offen'
#emoji_top_cluster = [['🤣', '😆', '😈'], ['😘', '👍', '💋', '😉'], ['🙄', '🤔', '😒', '😡'], ['💦'], ['💀', '💯', '💔'], ['😍', '❤️'], ['😭','😂']]#hate
emoji_top_cluster = [['😡', '💔', '😤', '🔥'], ['💗', '❤️'], ['🤣', '🙄', '🤔'], ['😆', '😀', '😬', '😄'], ['👍', '🙏'], ['🥺'], ['😱', '😳'], ['😭', '😂']]#offen

USER = re.compile(r"@\S+")
data_emoji = []
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
        for idx in range(len(emoji_top_cluster)):
            if emoji_one in emoji_top_cluster[idx]:
                user_tmp = USER.findall(line_clean)
                data_tmp = line_clean[:]
                for user in user_tmp:
                    data_tmp = data_tmp.replace(user + ' ', '')
                if len(data_tmp.strip().split(' ')) > 5:
                    data_emoji.append({'label': idx, 'text': line_clean})

random.shuffle(data_emoji)
SP = int(len(data_emoji)/10)
os.mkdir(TASK+'_cluster')
with open(TASK+'_cluster/train.json', 'a', encoding='utf-8') as f:
    for tweet in data_emoji[:9*SP]:
        json.dump(tweet, f)
        f.write('\n')
with open(TASK+'_cluster/dev.json', 'a', encoding='utf-8') as f:
    for tweet in data_emoji[9*SP:]:
        json.dump(tweet, f)
        f.write('\n')