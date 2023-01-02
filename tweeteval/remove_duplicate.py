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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='eval-irony,eval-hate,eval-offensive,eval-emotion,eval-stance',type=str)
parser.add_argument('--num',default=3,type=int)
parser.add_argument('--thre',default=0.95,type=float)
parser.add_argument('--print',default=1,type=int)
parser.add_argument('--write',default=1,type=int)
parser.add_argument('--length',default=3,type=int)
args = parser.parse_args()

def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            # t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            # t = 'http' if t.startswith('http') else t
            if t[0] == '@' and t.count('@') == 1:
                t = ''
            elif t[0] == '#' and t.count('#') == 1:
                t = ''
            elif t.startswith('http'):
                t = ''
            else:
                t = t
        if len(t) > 1:
            preprocessed_text.append(t)
    if len(preprocessed_text) < args.length:
        return ''
    else:
        return ' '.join(preprocessed_text)

######## USING ORIGINAL SPLITS
print('loading model')
from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()


for dataset_one in args.dataset.split(','):
    print('loading data')
    bad_pair = [[]]
    data_sem = []
    for sp in ['train','dev','test']:
        with open('../finetune/data/' + dataset_one + '/' + sp + '.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                one = json.loads(line)
                data_sem.append(one)
            bad_pair[0].append(len(data_sem))

    bad_idx = []
    fea_sem = torch.tensor([[]]).view(-1,768).cuda()
    for idx in trange(len(data_sem)):
        one = data_sem[idx]
        one = preprocess(one['text'])
        if len(one) < 3:
            bad_idx.append(idx)
            bad_pair.append([idx])
            if args.print == 1:
                print('empty_sample:'+data_sem[idx]['labels'] + ' ' +data_sem[idx]['text'])
        inputs = tokenizer(one, truncation=True)
        with torch.no_grad():
            outputs = model(input_ids=torch.tensor([inputs['input_ids']]).cuda(),
                            attention_mask=torch.tensor([inputs['attention_mask']]).cuda(),
                            # token_type_ids=torch.tensor([inputs['token_type_ids']]).cuda(),
                            output_hidden_states=True, return_dict=True).pooler_output
        fea_sem = torch.cat((fea_sem,outputs),0)


    for idx in range(len(data_sem)-1):
        if idx in bad_idx:
            continue
        fea_one = fea_sem[idx]
        dis = cos_sim(fea_one, fea_sem[idx+1:])
        val, best_idx = dis.topk(min(args.num,len(dis)))
        for val1,idx1 in zip(val,best_idx):
            val1 = val1.item()
            idx1 = idx1.item()
            if round(val1,2)  >= args.thre:
                if args.print == 1:
                    print('bad_pair_val:'+str(val1))
                    print('bad_pair1:'+data_sem[idx]['labels'] + ' ' +data_sem[idx]['text'])
                    print('bad_pair2:'+data_sem[idx1+idx+1]['labels'] + ' ' +data_sem[idx1+idx+1]['text'])
                    ### manual
                    # while 1:
                    #     rate = input('type your rate: ')
                    #     if rate == '1' or rate == '0':
                    #         break
                    # rate = int(rate)
                    rate = 1
                    if rate == 1:
                        bad_idx.append(idx1+idx+1)
                        bad_pair.append([idx,idx1+idx+1])
                        fea_sem[idx1+idx+1] = fea_sem[idx1+idx+1]*0

    with open('../finetune/data/' + dataset_one + '/bad_pair.txt', 'w', encoding='utf-8') as f:
        for line in bad_pair:
            line = [str(i) for i in line]
            f.write(' '.join(line) + '\n')

    bad_train = 0
    conflict_train = 0
    emp_train = 0
    bad_dev = 0
    conflict_dev = 0
    emp_dev = 0
    bad_test = 0
    conflict_test = 0
    emp_test = 0
    for pair in bad_pair:
        if len(pair) == 3: # record train dev test splits
            continue
        elif len(pair) == 1: # bad empty text
            if pair[0] < bad_pair[0][0]:
                emp_train += 1
            elif pair[0] < bad_pair[0][1]:
                emp_dev += 1
            else:
                emp_test += 1
        else:
            if pair[0] > bad_pair[0][0]: # only consider duplicate/conflict with train set
                continue
            if pair[1] < bad_pair[0][0]: # train
                bad_train += 1
                if data_sem[pair[0]]['labels'] != data_sem[pair[1]]['labels']:
                    conflict_train += 1
            elif pair[1] < bad_pair[0][1]: # dev
                bad_dev += 1
                if data_sem[pair[0]]['labels'] != data_sem[pair[1]]['labels']:
                    conflict_dev += 1
            elif pair[1] < bad_pair[0][2]: # test
                bad_test += 1
                if data_sem[pair[0]]['labels'] != data_sem[pair[1]]['labels']:
                    conflict_test += 1

    print('data:{}, train total:{}, empty:{}, duplicate:{}, conflict:{}'.format(dataset_one,bad_pair[0][0],emp_train,bad_train,conflict_train))
    print('data:{}, dev total:{} empty:{}, duplicate:{}, conflict:{}'.format(dataset_one,bad_pair[0][1]-bad_pair[0][0],emp_dev,bad_dev,conflict_dev))
    print('data:{}, test total:{} empty:{}, duplicate:{}, conflict:{}'.format(dataset_one,bad_pair[0][2]-bad_pair[0][1],emp_test,bad_test,conflict_test))

    if args.write == 1:
        if not os.path.isdir('../finetune/data/' + dataset_one + '_clean/'):
            os.mkdir('../finetune/data/' + dataset_one + '_clean/')
        with open('../finetune/data/' + dataset_one + '_clean/all.json', 'w', encoding='utf-8') as f:
            for idx in range(len(data_sem)):
                if idx not in bad_idx:
                    tmp = json.dumps(data_sem[idx], ensure_ascii=False)
                    f.write(tmp + '\n')
