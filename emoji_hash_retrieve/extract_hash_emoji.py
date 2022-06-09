import re
import json
import emoji
import argparse
from tqdm import tqdm,trange
parser = argparse.ArgumentParser()
parser.add_argument('task', type=str, default=None, required=True)

def write_json(fileName,data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

def main(args):
    with open('/work/data/twitter_hash.txt','r') as f:
        lines = f.readlines()
    for task in args.task.split(','):
        for sp in ['train','dev','test']:
            data = []
            with open('../finetune/data/'+task+'/'+sp+'_emo_hash.json') as f:
                for line in f:
                    one  = json.loads(line)
                    data.append(one)

            for one in tqdm(data):
                one['retrieve'] = []
                for line in lines:
                    hash_flag = 0
                    for hash_one in one['hash']:
                        if hash_one.lower() in line.lower():
                            hash_flag = 1
                            break
                    emoji_flag = 0
                    if hash_flag == 1:
                        for emoji_one in one['emoji']:
                            if emoji_one in line:
                                emoji_flag = 1
                                break
                    if hash_flag ==1 and emoji_flag == 1:
                        one['retrieve'].append(line)

            write_json('../finetune/data/'+task+'/'+sp+'_emo_hash_retrieve.json', data)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)