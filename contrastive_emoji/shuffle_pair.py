import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100000,type=int)
parser.add_argument('--num',default=1000,type=int)
args = parser.parse_args()

import json
from tqdm import tqdm
data = []
with open('emoji_pair_thre'+str(args.thre)+'_num'+str(args.num), 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        one = json.loads(line)
        data.append(one)

import random
random.shuffle(data)

def write_json(fileName,data):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

write_json('emoji_pair_thre'+str(args.thre)+'_num'+str(args.num)+'_shuffle', data)