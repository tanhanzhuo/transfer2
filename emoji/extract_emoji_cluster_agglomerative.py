import random
import json
random.seed(0)
from tqdm import tqdm,trange
# emoji_top = {'👀':0, '🙌':0, '😳':0, '💀':0, '😔':0, '😏':0, '😁':0, '❤':0, '🔥':0, '👌':0,
#         '😒':0, '💯':0, '💕':0, '😘':0, '😊':0, '😩':0, '❤️':0, '😍':0, '😭':0, '😂':0}
emoji_top = ['🤔', '🙄', '😳', '👌', '😁', '😏', '💯', '🔥', '💕', '😘', '😔', '❤', '♥', '😒', '😊', '😩', '❤️', '😍', '😭', '😂']
emoji_top_cluster = [['😁', '😏', '💕', '😘', '😊', '😍'], ['👌', '💯', '🔥'], ['😳', '😔', '😒', '😩'], ['❤', '♥', '❤️'], ['🤔', '🙄'], ['😭', '😂']]

label2idx = {}
for idx in range(len(emoji_top)):
    label2idx[emoji_top[idx]] = idx
with open('data_emoji.txt', 'r') as f:
    data_emoji = f.readlines()

data_emoji_top = []
for data_one in tqdm(data_emoji[:100000]):
    emoji_one = data_one.split('\t')[0]
    line = data_one.split('\t')[1].strip().replace('  ',' ')
    if emoji_one in emoji_top:
        if len(line.split(' ')) > 5:
            txt = line.replace('https://', 'https') + '\n'
            lab = -1
            for idx_lab in range(len(emoji_top_cluster)):
                if emoji_one in emoji_top_cluster[idx_lab]:
                    lab = idx_lab
            if lab == -1:
                print('****************ERROR***************')
            data_emoji_top.append(
                {'label': lab, 'text': txt}
            )

random.shuffle(data_emoji_top)
SP = int(len(data_emoji_top)*0.9)
with open('./cluster_train/train.json', 'w') as f:
    for idx in trange(SP):
        json.dump(data_emoji_top[idx], f)
        f.write('\n')
with open('./cluster_train/dev.json', 'w') as f:
    for idx in trange(SP,len(data_emoji_top)):
        json.dump(data_emoji_top[idx], f)
        f.write('\n')