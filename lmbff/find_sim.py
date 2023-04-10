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
parser.add_argument('--dataset',default='sem21-task7-humor,sem22-task6-sarcasm,stance,eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate',type=str)
parser.add_argument('--num',default=1,type=int)
parser.add_argument('--thre',default=0.95,type=float)
parser.add_argument('--print',default=0,type=int)
parser.add_argument('--write',default=0,type=int)
parser.add_argument('--write_demo',default=0,type=int)
parser.add_argument('--length',default=1,type=int)
args = parser.parse_args()

CONVERT = {
    'eval-emoji':{'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,\
                  '9':9,'10':10,'11':11,'12':12,'13':13,'14':14,'15':15,\
                  '16':16,'17':17,'18':18,'19':19},
    'eval-emotion':{'0':0,'1':1,'2':2,'3':3},
    'eval-hate':{'0':0,'1':1},
    'eval-irony':{'0':0,'1':1},
    'eval-offensive':{'0':0,'1':1},
    'eval-sentiment':{'0':0,'1':1,'2':2},
    'eval-stance/abortion':{'0':0,'1':1,'2':2},
    'eval-stance/atheism':{'0':0,'1':1,'2':2},
    'eval-stance/climate':{'0':0,'1':1,'2':2},
    'eval-stance/feminist':{'0':0,'1':1,'2':2},
    'eval-stance/hillary':{'0':0,'1':1,'2':2},
    'eval-sarcasm': {'0': 0, '1': 1},
    'eval-stance': {'0': 0, '1': 1, '2': 2},
    'stance': {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}

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
    data_sem = {}
    for sp in ['train','dev','test']:
        data_sem[sp] = []
        with open('../finetune/data/' + dataset_one + '/' + sp + '.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                one = json.loads(line)
                data_sem[sp].append(one)

        label_split = []
        for tmp in range(len(CONVERT[dataset_one].keys())):
            label_split.append([])
        if sp == 'train':
            for idx in range(len(data_sem[sp])):
                one = data_sem[sp]
                lab = CONVERT[dataset_one][one['label']]
                label_split[lab].append(idx)

        random.shuffle(data_sem[sp])
        fea_sem = torch.tensor([[]]).view(-1,768).cuda()
        for idx in trange(len(data_sem[sp])):
            one = data_sem[sp][idx]
            one = preprocess(one['text'])
            inputs = tokenizer(one, truncation=True)
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([inputs['input_ids']]).cuda(),
                                attention_mask=torch.tensor([inputs['attention_mask']]).cuda(),
                                # token_type_ids=torch.tensor([inputs['token_type_ids']]).cuda(),
                                output_hidden_states=True, return_dict=True).pooler_output
            fea_sem = torch.cat((fea_sem,outputs),0)
        data_sem[sp+'_token'] = copy.deepcopy(fea_sem)

    fea_sem = copy.deepcopy(data_sem['train_token'])
    data_train  = copy.deepcopy(data_sem['train'])
    for idx in range(len(data_sem['train_token'])):
        fea_one = data_sem['train_token'][idx]
        dis = cos_sim(fea_one, fea_sem)
        dis[idx] = 0
        for idx2 in range(len(label_split)):
            dis_tmp = dis[label_split[idx2]]
            idx_best = torch.argmax(dis_tmp)
            data_train[idx]['text'+str(idx2)] = data_sem['train'][ label_split[idx2][idx_best] ]['text']

    data_dev = copy.deepcopy(data_sem['dev'])
    for idx in range(len(data_sem['dev_token'])):
        fea_one = data_sem['dev_token'][idx]
        dis = cos_sim(fea_one, fea_sem)
        for idx2 in range(len(label_split)):
            dis_tmp = dis[label_split[idx2]]
            idx_best = torch.argmax(dis_tmp)
            data_dev[idx]['text' + str(idx2)] = data_sem['train'][label_split[idx2][idx_best]]['text']

    data_test = copy.deepcopy(data_sem['test'])
    for idx in range(len(data_sem['test_token'])):
        fea_one = data_sem['test_token'][idx]
        dis = cos_sim(fea_one, fea_sem)
        for idx2 in range(len(label_split)):
            dis_tmp = dis[label_split[idx2]]
            idx_best = torch.argmax(dis_tmp)
            data_test[idx]['text' + str(idx2)] = data_sem['train'][label_split[idx2][idx_best]]['text']

    if args.write_demo == 1:
        if not os.path.isdir('../finetune/data/' + dataset_one + '_demo/'):
            os.mkdir('../finetune/data/' + dataset_one + '_demo/')
        with open('../finetune/data/' + dataset_one + '_demo/train.json', 'w', encoding='utf-8') as f:
            for idx in range(len(data_sem['train'])):
                if idx not in emp_idx and idx not in bad_idx_flat:
                    tmp = json.dumps(data_sem['train'][idx], ensure_ascii=False)
                    f.write(tmp + '\n')
        import shutil
        shutil.copyfile('../finetune/data/' + dataset_one + '/dev.json', '../finetune/data/' + dataset_one + '_demo/dev.json')
        shutil.copyfile('../finetune/data/' + dataset_one + '/test.json',
                        '../finetune/data/' + dataset_one + '_demo/test.json')
        # with open('../finetune/data/' + dataset_one + '_demo/dev.json', 'w', encoding='utf-8') as f:
        #     for idx in range(len(data_sem['train'])):
        #         if idx not in bad_idx:
        #             tmp = json.dumps(data_sem[idx], ensure_ascii=False)
        #             f.write(tmp + '\n')
        # with open('../finetune/data/' + dataset_one + '_demo/test.json', 'w', encoding='utf-8') as f:
        #     for idx in range(bad_pair[0][1],bad_pair[0][2]):
        #         if idx not in bad_idx:
        #             tmp = json.dumps(data_sem[idx], ensure_ascii=False)
        #             f.write(tmp + '\n')
