import json
import re
import emoji
import numpy as np
import random
from transformers import AutoTokenizer

token = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
HASH = re.compile(r"#\S+")
MIN = 2
def write_json(fileName,data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

def read_data(task,sp,KTH):
    data = []
    with open('../finetune/data/' + task + '/' + sp + '_emo_hash_retrieve.json') as f:
        for line in f:
            one = json.loads(line)
            hash_one = one['hash']
            emoji_one = one['emoji']
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
            one['retrieve'] = []
            for idx in best:
                text = retrieve_all[idx].replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]',
                                                                                                'https').strip()
                one['retrieve'].append(text)
            data.append(one)
    return data

for task in 'sem-18,sem19-task6-offen'.split(','):
    for KTH in [2]:
        data_train = []
        for sp in ['train','dev']:
            data_train.extend(read_data(task,sp,KTH))
        if KTH == 0:
            idx_perm = list(range(len(data_train)))
            random.shuffle(idx_perm)
        NUM = int(len(data_train)*0.1)
        write_json('../finetune/data/'+task+'/'+'train'+'_emo_hash_process_semi'+str(KTH)+'.json', [ data_train[i] for i in idx_perm[:NUM*9] ])
        write_json('../finetune/data/' + task + '/' + 'dev' + '_emo_hash_process_semi'+str(KTH)+'.json', [ data_train[i] for i in idx_perm[NUM*9:] ])
        data_test = read_data(task,'test',KTH)
        write_json('../finetune/data/' + task + '/' + 'test' + '_emo_hash_process_semi'+str(KTH)+'.json', data_test)

