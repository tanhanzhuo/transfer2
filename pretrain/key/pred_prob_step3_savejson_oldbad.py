import pickle
import emoji
from tqdm import tqdm,trange
import re
prob_map = []
for split in range(1):
    with open('/work/test/pretrain_hashtag/prob' + '_sep8_' + str(split) + '.pickle', 'rb') as handle:
        tmp = pickle.load(handle)
        tmp = [list(tt) for tt in tmp]
        print(len(tmp))
        prob_map.extend(tmp)

# for split in range(4):
#     with open('/work/test/pretrain_hashtag/prob' + '_makeup_' + str(split) + '.pickle', 'rb') as handle:
#         tmp = pickle.load(handle)
#         tmp = [list(tt) for tt in tmp]
#         print(len(tmp))
#         prob_map.extend(tmp)
import datasets
train_dataset = datasets.load_from_disk('/work/test/pretrain_hashtag/twitter_ref_clean_simple/TrainData_line')["train"]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)

import json
with open('/work/test/pretrain_hashtag/txt_prob_200m_new2.json', 'w') as f:
    for idx in trange(len(prob_map)):
        one = train_dataset[idx]['input_ids'][1:-1]
        if len(one) <= 3:
            continue
        txt = tokenizer.decode(one)
        
        txt = emoji.emojize(txt)
        

        prob = prob_map[idx]
        out_dic = {'text':txt,'prob':prob}
        out_dic = json.dumps(out_dic, ensure_ascii=False)
        f.write(out_dic+'\n')