import json
import re
import emoji
import numpy as np
from transformers import AutoTokenizer
def write_json(fileName,data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')
token = AutoTokenizer.from_pretrained('vinai/bertweet-base')
KTH = 2
HASH = re.compile(r"#\S+")
for task in 'sem-18,sem19-task6-offen'.split(','):
    for sp in ['train','dev','test']:
        data = []
        with open('../finetune/data/'+task+'/'+sp+'_emo_hash_retrieve.json') as f:
            for line in f:
                one  = json.loads(line)
                hash_one = one['hash']
                emoji_one = one['emoji']
                similarity = []
                retrieve_all = one.pop('retrieve')
                if len(retrieve_all) < KTH:
                    continue
                for retrieve in retrieve_all:
                    hash_re = HASH.findall(retrieve)
                    emoji_re = emoji.distinct_emoji_list(retrieve)
                    hash_same = list(set(hash_re) & set(hash_one))
                    emoji_same = list(set(emoji_re) & set(emoji_one))
                    similarity.append(len(hash_same)+len(emoji_same))
                best = np.argpartition(np.array(similarity), -KTH)[-KTH:]

                for idx in best:
                    one['text'] = retrieve_all[idx].replace('[RT]', '').replace('[USER]', '@USER').replace('[HTTP]', 'https').strip()\
                                  + ' '+token.eos_token + ' '\
                                  + one['text']
                data.append(one)
        write_json('../finetune/data/'+task+'/'+sp+'_emo_hash_process.json', data)
