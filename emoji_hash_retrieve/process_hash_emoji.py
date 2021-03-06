import json
import re
import emoji
import numpy as np
import random
from transformers import AutoTokenizer,set_seed
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', type=str, default='_emo_hash_process_orifirst')
parser.add_argument('--order', type=str, default='ori')
token = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
HASH = re.compile(r"#\S+")
MIN = 2

def write_json(fileName,data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

def read_data(task,sp,KTH,order):
    data = []
    with open('../finetune/data/' + task + '/' + sp + '_emo_hash_retrieve.json') as f:
        for line in f:
            one = json.loads(line)
            hash_one = one.pop('hash')
            emoji_one = one.pop('emoji')
            similarity = []
            retrieve_all = one.pop('retrieve')
            if len(retrieve_all) < MIN:
                continue
            if KTH == 0:
                data.append(one)
                continue

            for retrieve in retrieve_all:
                hash_re = HASH.findall(retrieve)
                emoji_re = emoji.distinct_emoji_list(retrieve)
                hash_same = list(set(hash_re) & set(hash_one))
                emoji_same = list(set(emoji_re) & set(emoji_one))
                similarity.append(len(hash_same) + len(emoji_same))
            best = np.argpartition(np.array(similarity), -KTH)[-KTH:]

            for idx in best:
                text = retrieve_all[idx].replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]',
                                                                                                'https').strip()
                if len(token(text)['input_ids']) > 30:
                    emoji_re = emoji.distinct_emoji_list(text)
                    emoji_same = list(set(emoji_re) & set(emoji_one))
                    text = emoji.replace_emoji(text) + ''.join(emoji_same)

                    hash_re = HASH.findall(text)
                    hash_same = list(set(hash_re) & set(hash_one))
                    for hash_tmp in hash_re:
                        if hash_tmp not in hash_same:
                            text = text.replace(hash_tmp + ' ', '')
                    # print(len(token(text)['input_ids']))
                if order == 'retrieve':
                    one['text'] = text + ' ' + token.eos_token + ' ' + one['text']
                elif order == 'ori':
                    one['text'] = one['text'] + ' ' + token.eos_token + ' ' + text
                else:
                    print('error!!!!!!!!!!!')
            data.append(one)
    return data

def main(args):
    set_seed(args.seed)
    for task in 'sem-18,sem19-task6-offen'.split(','):
        for KTH in [0,1,2]:
            data_train = []
            for sp in ['train','dev']:
                data_train.extend(read_data(task,sp,KTH,args.order))
            if KTH == 0:
                idx_perm = list(range(len(data_train)))
                random.shuffle(idx_perm)
            NUM = int(len(data_train)*0.1)
            write_json('../finetune/data/'+task+'/'+'train'+args.name+str(KTH)+'.json', [ data_train[i] for i in idx_perm[:NUM*9] ])
            write_json('../finetune/data/' + task + '/' + 'dev' + args.name+str(KTH)+'.json', [ data_train[i] for i in idx_perm[NUM*9:] ])
            data_test = read_data(task,'test',KTH,args.order)
            write_json('../finetune/data/' + task + '/' + 'test' + args.name+str(KTH)+'.json', data_test)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)