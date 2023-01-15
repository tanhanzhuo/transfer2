import json
import random

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
#'eval-irony,eval-hate,eval-offensive,eval-emotion,eval-stance'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance',type=str)
parser.add_argument('--num',default=3,type=int)
parser.add_argument('--thre',default=0.97,type=float)
parser.add_argument('--print',default=1,type=int)
parser.add_argument('--write',default=0,type=int)
parser.add_argument('--write_clean',default=1,type=int)
parser.add_argument('--length',default=1,type=int)
args = parser.parse_args()

def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            # t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            # t = 'http' if t.startswith('http') else t
            if t[0] == '@' and t.count('@') == 1:
                t = ''
            # elif t[0] == '#' and t.count('#') == 1:
            #     t = ''
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
    emp_idx = []
    data_sem = {}
    for sp in ['train','dev','test']:
        data_sem[sp] = []
        with open('../finetune/data/' + dataset_one + '/' + sp + '.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                one = json.loads(line)
                data_sem[sp].append(one)
        random.shuffle(data_sem[sp])
        fea_sem = torch.tensor([[]]).view(-1,768).cuda()
        for idx in trange(len(data_sem[sp])):
            one = data_sem[sp][idx]
            one = preprocess(one['text'])
            if len(one) < 3 and sp == 'train':
                if args.print == 1:
                    print('empty_sample:'+data_sem[idx]['labels'] + ' ' +data_sem[idx]['text'])
                emp_idx.append(idx)
            inputs = tokenizer(one, truncation=True)
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([inputs['input_ids']]).cuda(),
                                attention_mask=torch.tensor([inputs['attention_mask']]).cuda(),
                                # token_type_ids=torch.tensor([inputs['token_type_ids']]).cuda(),
                                output_hidden_states=True, return_dict=True).pooler_output
            fea_sem = torch.cat((fea_sem,outputs),0)
        data_sem[sp+'_token'] = copy.deepcopy(fea_sem)

    bad_idx = {'train':[],'dev':[],'test':[]}
    fea_sem = copy.deepcopy(data_sem['train_token'])
    for idx in range(len(data_sem['train_token'])-1):
        if idx in emp_idx or idx in bad_idx['train']:
            continue
        fea_one = data_sem['train_token'][idx]
        dis = cos_sim(fea_one, fea_sem[idx+1:])
        val, best_idx = dis.topk(min(args.num, len(dis)))
        for val1, idx1 in zip(val, best_idx):
            val1 = val1.item()
            idx1 = idx1.item()
            if round(val1, 2) >= args.thre:
                if args.print == 1:
                    print('bad_pair_val:' + str(val1))
                    print('bad_pair1:' + data_sem['train'][idx]['labels'] + ' ' + data_sem['train'][idx]['text'])
                    print('bad_pair2:' + data_sem['train'][idx1 + idx + 1]['labels'] + ' ' + data_sem['train'][idx1 + idx + 1]['text'])

                bad_idx['train'].append(idx1 + idx + 1)
                fea_sem[idx1 + idx + 1] = fea_sem[idx1 + idx + 1] * 0

    for sp in ['dev', 'test']:
        for idx in range(len(data_sem[sp+'_token'])):
            fea_one = data_sem[sp+'_token'][idx]
            dis = cos_sim(fea_one, data_sem['train_token'])
            val, best_idx = dis.topk(min(args.num,len(dis)))
            for val1,idx1 in zip(val,best_idx):
                val1 = val1.item()
                idx1 = idx1.item()
                if round(val1,2)  >= args.thre:
                    if args.print == 1:
                        print('bad_pair_val:'+str(val1))
                        print('bad_pair1:'+data_sem[sp][idx]['labels'] + ' ' +data_sem[sp][idx]['text'])
                        print('bad_pair2:'+data_sem['train'][idx1]['labels'] + ' ' +data_sem['train'][idx1]['text'])
                    bad_idx[sp].append(idx1)

    print('data:{}, train total:{}, empty:{}, duplicate:{}'.format(dataset_one,len(data_sem['train']),len(emp_idx),len(bad_idx['train'])))
    print('data:{}, dev total:{} empty:{}, duplicate:{}'.format(dataset_one,len(data_sem['dev']),len(emp_idx),len(bad_idx['dev'])))
    print('data:{}, test total:{} empty:{}, duplicate:{}'.format(dataset_one,len(data_sem['test']),len(emp_idx),len(bad_idx['test'])))

    bad_idx_flat = []
    for values in bad_idx.values():
        bad_idx_flat.extend(list(values))
    bad_idx_flat = list( set(bad_idx_flat) )
    if args.write_clean == 1:
        if not os.path.isdir('../finetune/data/' + dataset_one + '_clean/'):
            os.mkdir('../finetune/data/' + dataset_one + '_clean/')
        with open('../finetune/data/' + dataset_one + '_clean/train.json', 'w', encoding='utf-8') as f:
            for idx in range(len(data_sem['train'])):
                if idx not in emp_idx and idx not in bad_idx_flat:
                    tmp = json.dumps(data_sem['train'][idx], ensure_ascii=False)
                    f.write(tmp + '\n')
        # with open('../finetune/data/' + dataset_one + '_clean/dev.json', 'w', encoding='utf-8') as f:
        #     for idx in range(len(data_sem['train'])):
        #         if idx not in bad_idx:
        #             tmp = json.dumps(data_sem[idx], ensure_ascii=False)
        #             f.write(tmp + '\n')
        # with open('../finetune/data/' + dataset_one + '_clean/test.json', 'w', encoding='utf-8') as f:
        #     for idx in range(bad_pair[0][1],bad_pair[0][2]):
        #         if idx not in bad_idx:
        #             tmp = json.dumps(data_sem[idx], ensure_ascii=False)
        #             f.write(tmp + '\n')
