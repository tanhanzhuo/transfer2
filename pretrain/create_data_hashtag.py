import argparse
import numpy as np
import json
from tqdm import tqdm, trange
import re
import string
import random
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--thre', default=0.98, type=float)
parser.add_argument('--num', default=1000, type=int)
args = parser.parse_args()

import json
with open('../contrastive/hash_his.json','r',encoding='utf-8') as f:
    hash_dic = json.load(f)

HASH = re.compile(r"#\S+")
USER = re.compile(r"@\S+")
HTTP = re.compile(r"http\S+")
META = re.compile(r"[http|#|@]\S+")
filePath = '/work/data/twitter_hash_clean.txt' #'twitter_hash_test_clean.txt'#

def process(line):
    hash_tmp = HASH.findall(line)
    hash_tmp_clean = []
    for hash_one in hash_tmp:
        # hash_one = hash_one.lower()
        if len(hash_one) > 30:
            continue
        if hash_one[1].isalpha():
            if hash_one[-1] == '…':
                continue
            if len(hash_one) > 3 and hash_one[-3:] == '...':
                continue
            if hash_one[-1] in string.punctuation:
                hash_one = hash_one[:-1]
            hash_clean = re.findall('[a-zA-Z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                hash_tmp_clean.append(hash_one)

    return hash_tmp_clean


hash_thre_list = list(hash_dic.keys())
hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = set()
hash_bad = set()
with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        hash_tmp_clean = process(line)
        for hash_one in hash_tmp_clean:
            tmp = hash_data.get(hash_one.lower())
            if tmp is not None:
                hash_data[hash_one.lower()].add(line)


from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
import torch
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()


NUM = args.num
THRE=args.thre
for hash_one in tqdm(hash_thre_list):
    hash_data_one = list(hash_data[hash_one])
    random.shuffle(hash_data_one)
    if len(hash_data_one) > NUM*2:
        hash_data_one[:NUM*2]
    if len(hash_data_one) < 10:
        continue
    hash_data_one_remove = []
    bad_idx = []
    # fea_sem = torch.tensor([[]]).view(-1, 768).cuda()
    for idx in range(len(hash_data_one)):
        text_one = hash_data_one[idx]
        text_one = text_one.replace('@USER', '').replace('https', '').strip()
        hash_tmp = HASH.findall(text_one)
        for hash_tmp1 in hash_tmp:
            text_one = text_one.replace(hash_tmp1, '')
        if len(text_one.replace(' ','')) < 10:
            bad_idx.append(idx)
        hash_data_one_remove.append(text_one)

    hash_data_one_remove_token = tokenizer(
        hash_data_one_remove,
        padding=True,
        truncation=True,
        max_length=128,
        return_special_tokens_mask=False,
        return_token_type_ids=False
    )
    with torch.no_grad():
        fea_sem = model(input_ids=torch.tensor(hash_data_one_remove_token['input_ids']).cuda(),
                        attention_mask=torch.tensor(hash_data_one_remove_token['attention_mask']).cuda(),
                        # token_type_ids=torch.tensor([inputs['token_type_ids']]).cuda(),
                        output_hidden_states=True, return_dict=True).pooler_output
    # fea_sem = torch.cat((fea_sem, outputs), 0)
    # fea_sem = copy.deepcopy(outputs)
    for idx in range(len(fea_sem)-1):
        if idx in bad_idx:
            continue
        fea_one = fea_sem[idx]
        dis = cos_sim(fea_one, fea_sem[idx+1:])
        for idx1 in range(len(dis)):
            dis_one = dis[idx1]
            if dis_one > THRE:
                bad_idx.append(idx1 + idx + 1)
                fea_sem[idx1 + idx + 1] = fea_sem[idx1 + idx + 1] * 0

    with open('tweet_hash_clean_group.txt', 'a', encoding='utf-8') as f:
        f.write('TANS_HASH:'+hash_one+'\n')
        num_count = 0
        for idx in range(len(fea_sem)):
            if idx not in bad_idx:
                f.write(hash_data_one[idx])
                num_count += 1
                if num_count > NUM:
                    break

    # del outputs
    # torch.cuda.empty_cache()



