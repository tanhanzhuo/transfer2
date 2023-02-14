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
parser.add_argument('--tasks',default='eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean',type=str)
parser.add_argument('--target',default='_clean',type=str)
parser.add_argument('--method',default='_fulldata_simcse_top20_textfirst',type=str)#_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst''

parser.add_argument('--model',default='1399999',type=str)
parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

args = parser.parse_args()

from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
from models import RobertaForCL
from torch.utils.data import DataLoader
config = AutoConfig.from_pretrained(args.model)
# model = AutoModel.from_pretrained(args.model,config=config).cuda()
model = RobertaForCL.from_pretrained(
                args.model,
                config=config,
                model_args=args
            ).cuda()
model.eval()
# model = accelerate.prepare(model)
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()


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

                data_source_text = one['text'].split(' \n ')[:-1]

                inputs = tokenizer(one['text'].split(' \n ')[:-1], truncation=True, max_length=128, padding=True)
                with torch.no_grad():
                    outputs = model(input_ids=torch.tensor(inputs['input_ids']).cuda(),
                                    attention_mask=torch.tensor(inputs['attention_mask']).cuda(),
                                    output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
                    dis = cos_sim(outputs[-1], outputs[:-1])
                    val, best_idx = dis.topk(len(dis))
                    text_rerank = one['text'].split(' \n ')[-2]
                    for cur_idx in best_idx.flip(0):
                        text_rerank = data_source_text[cur_idx].strip() \
                                                     + ' \n ' + text_rerank.strip() + ' \n '
                    data_source.append({'labels':one['labels'],'text':text_rerank})
        write_json(data_source, '../finetune/data/' + task + '/' + sp + args.method + '_rerank')