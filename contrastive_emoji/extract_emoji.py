import random
import json
random.seed(0)
# emoji_top = {'👀':0, '🙌':0, '😳':0, '💀':0, '😔':0, '😏':0, '😁':0, '❤':0, '🔥':0, '👌':0,
#         '😒':0, '💯':0, '💕':0, '😘':0, '😊':0, '😩':0, '❤️':0, '😍':0, '😭':0, '😂':0}
emoji_top = ['🤔', '🙄', '😳', '👌', '😁', '😏', '💯', '🔥', '💕', '😘', '😔', '❤', '♥', '😒', '😊', '😩', '❤️', '😍', '😭', '😂']
label2idx = {}
for idx in range(len(emoji_top)):
    label2idx[emoji_top[idx]] = idx
with open('data_emoji.txt', 'r') as f:
    data_emoji = f.readlines()

data_emoji_top = []
for data_one in data_emoji:
    emoji_one = data_one.split('\t')[0]
    line = data_one.split('\t')[1].strip().replace('  ',' ')
    if emoji_one in emoji_top:
        if len(line.split(' ')) > 5:
            txt = line.replace('https://', 'https') + '\n'
            lab = label2idx[emoji_one]
            data_emoji_top.append(
                {'label': lab, 'text': txt}
            )

random.shuffle(data_emoji_top)
SP = int(len(data_emoji_top)*0.9)
with open('train.json', 'w') as f:
    for idx in range(SP):
        json.dump(data_emoji_top[idx], f)
        f.write('\n')
with open('dev.json', 'w') as f:
    for idx in range(SP,len(data_emoji_top)):
        json.dump(data_emoji_top[idx], f)
        f.write('\n')