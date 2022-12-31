import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import os
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='eval-irony',type=str)
parser.add_argument('--num',default=1,type=int)
parser.add_argument('--thre',default=0.90,type=float)
parser.add_argument('--print',default=1,type=int)
args = parser.parse_args()

def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            # t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            # t = 'http' if t.startswith('http') else t
            t = '' if t[0] == '@' and t.count('@') == 1 else t
            t = '' if t.startswith('http') else t
        preprocessed_text.append(t)
    return ' '.join(preprocessed_text)

######## USING ORIGINAL SPLITS
print('loading model')
from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()

print('loading data')



data_sem = []
for sp in ['train','dev','test']:
    with open('../finetune/data/' + args.dataset + '/' + sp + '.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            one = json.loads(line)
            data_sem.append(one)

fea_sem = torch.tensor([[]]).view(-1,768).cuda()
for one in tqdm(data_sem):
    one = preprocess(one['text'])
    input = tokenizer(one, truncation=True)
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
                        attention_mask=torch.tensor([input['attention_mask']]).cuda(),
                        # token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
                        output_hidden_states=True, return_dict=True).pooler_output
    fea_sem = torch.cat((fea_sem,outputs),0)


total_du = 0
for idx in trange(len(data_sem)-1):
    fea_one = fea_sem[idx]
    dis = cos_sim(fea_one, fea_sem[idx+1:])
    val, best_idx = dis.topk(args.num)
    if val[0]  > args.thre:
        for idx2 in range(args.num):
            if args.print == 1:
                print(val[idx2])
                print(data_sem[idx]['text'])
                print(data_sem[best_idx[idx2+idx+1]]['text'])
        total_du += 1
print('duplicate:{}/{}'.format(total_du,len(data_sem)))
