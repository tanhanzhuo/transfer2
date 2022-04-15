import random
random.seed(0)
# emoji_top = {'👀':0, '🙌':0, '😳':0, '💀':0, '😔':0, '😏':0, '😁':0, '❤':0, '🔥':0, '👌':0,
#         '😒':0, '💯':0, '💕':0, '😘':0, '😊':0, '😩':0, '❤️':0, '😍':0, '😭':0, '😂':0}
emoji_top = ['🤔', '🙄', '😳', '👌', '😁', '😏', '💯', '🔥', '💕', '😘', '😔', '❤', '♥', '😒', '😊', '😩', '❤️', '😍', '😭', '😂']
with open('data_emoji.txt', 'r') as f:
    data_emoji = f.readlines()

data_emoji_top = []
for data_one in data_emoji:
    emoji_one = data_one.split('\t')[0]
    line = data_one.split('\t')[1].strip().replace('  ',' ')
    if emoji_one in emoji_top:
        if len(line.split(' ')) > 5:
            data_emoji_top.append(data_one.replace('https://','https'))
random.shuffle(data_emoji_top)
SP = int(len(data_emoji_top)*0.9)
with open('data_emoji_train', 'w') as f:
    for idx in range(SP):
        f.write(data_emoji_top[idx])
with open('data_emoji_test', 'w') as f:
    for idx in range(SP,len(data_emoji_top)):
        f.write(data_emoji_top[idx])