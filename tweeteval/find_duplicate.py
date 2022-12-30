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
            t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            t = 'http' if t.startswith('http') else t
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


def read_data(args,sp):
    data_sem = []
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
    return data_sem,fea_sem

data_train, fea_train = read_data(args, 'train')
data_dev, fea_dev = read_data(args, 'dev')
data_test, fea_test = read_data(args, 'test')

total_dev = 0
for idx in trange(len(data_dev)):
    fea_one = fea_dev[idx]
    dis = cos_sim(fea_one, fea_train)
    val, best_idx = dis.topk(args.num)
    if val[0]  > args.thre:
        for idx2 in range(args.num):
            if args.print == 1:
                print(val[idx2])
                print(data_dev[idx]['text'])
                print(data_train[best_idx[idx2]]['text'])
        total_dev += 1
print('dev:{}/{}'.format(total_dev,len(data_dev)))

total_test = 0
for idx in trange(len(data_test)):
    fea_one = fea_test[idx]
    dis = cos_sim(fea_one, fea_train)
    val, best_idx = dis.topk(args.num)
    if val[0]  > args.thre:
        for idx2 in range(args.num):
            if args.print == 1:
                print(val[idx2])
                print(data_test[idx]['text'] +'-----'+ str(data_test[idx]['labels']))
                print(data_train[best_idx[idx2]]['text'] +'-----'+ str(data_train[best_idx[idx2]]['labels']))
        total_test += 1
print('test:{}/{}'.format(total_test,len(data_test)))