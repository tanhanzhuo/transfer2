emoji_top = {'👀':0, '🙌':0, '😳':0, '💀':0, '😔':0, '😏':0, '😁':0, '❤':0, '🔥':0, '👌':0,
        '😒':0, '💯':0, '💕':0, '😘':0, '😊':0, '😩':0, '❤️':0, '😍':0, '😭':0, '😂':0}

import emoji
from tqdm import tqdm

data_emoji = []
with open('/work/data/twitter_ref.txt', 'r') as f:
    for line in tqdm(f):
        emoji_list = emoji.distinct_emoji_list(line)
        if len(emoji_list) ==1:
            if emoji_list[0] in emoji_top.keys():
                emoji_top[emoji_list[0]] += 1
                data_emoji.append(line)
with open('data_emoji.txt', 'w') as f:
    for line in data_emoji:
        f.write(line)
print(emoji_top)
