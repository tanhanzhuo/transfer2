import json
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one['stance'] + '\t' + one['text'].replace('\t', '').replace('\n', ' ') + '\n')
with open('../data_raw/wtwt/wtwt_with_text.json', 'r') as f:
    data = json.load(f)
random.shuffle(data)
SP = int(len(data)/10)
write_txt('./wtwt/train', data[:8*SP])
write_txt('./wtwt/dev', data[8*SP:9*SP])
write_txt('./wtwt/test', data[9*SP:])