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
from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--tasks',default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm',type=str)
parser.add_argument('--target',default='_clean',type=str)
parser.add_argument('--method',default='_seg_500_one20_top100_sp',type=str)#'_fulldata_simcse_top20_textfirst'
parser.add_argument('--num',default=100,type=int)
args = parser.parse_args()

def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

for task in args.tasks.split(','):
    print(task)
    for sp in ['train', 'dev', 'test']:
        data_source = []
        with open('../finetune/data/' + task + '/' + sp + args.method + '.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                one = json.loads(line)
                text = [one['text']]
                for idx_tmp in range(args.num):
                    text.append(one['text'+str(idx_tmp)])
                inputs = tokenizer(text, truncation=True, max_length=128, padding=True)
                with torch.no_grad():
                    outputs = model(input_ids=torch.tensor(inputs['input_ids']).cuda(),
                                    attention_mask=torch.tensor(inputs['attention_mask']).cuda(),
                                    # token_type_ids=torch.tensor([inputs['token_type_ids']]).cuda(),
                                    output_hidden_states=True, return_dict=True).pooler_output
                    dis = cos_sim(outputs[0], outputs[1:])
                    val, best_idx = dis.topk(len(dis))
                    data_one = {'labels':one['labels'],'text':one['text']}
                    for idx_tmp in range(len(best_idx)):
                        data_one['text'+str(idx_tmp)] = one['text'+str(best_idx[idx_tmp].item())]
                    data_source.append(data_one)
        write_json(data_source, '../finetune/data/' + task + '/' + sp + args.method + '_rerank')