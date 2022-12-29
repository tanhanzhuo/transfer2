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
parser.add_argument('--tweeteval',default='eval-irony_evensplit4',type=str)
parser.add_argument('--semeval',default='sem-18',type=str)
parser.add_argument('--method',default='modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst',type=str)
args = parser.parse_args()

def preprocess(text):
    preprocessed_text = []
    for t in text.split():
        if len(t) > 1:
            t = '@user' if t[0] == '@' and t.count('@') == 1 else t
            t = 'http' if t.startswith('http') else t
        preprocessed_text.append(t)
    return ' '.join(preprocessed_text)

# task = args.tweeteval
# data_tweet = []
# for sp in ['train','val','test']:
#     with open('../finetune/data_raw/eval-'+task+'/'+sp+'_text.txt','r',encoding='utf-8') as f:
#         lines_text = f.readlines()
#     with open('../finetune/data_raw/eval-' + task + '/' + sp + '_labels.txt', 'r', encoding='utf-8') as f:
#         lines_labels = f.readlines()
#
#     for idx in range(len(lines_text)):
#         line_text = preprocess(lines_text[idx].strip())
#         line_label = lines_labels[idx].strip()
#         if len(line_text)<1:
#             continue
#         one = {'labels':line_label,'text':line_text}
#         data_tweet.append(one)
#
# data_sem = []
# for sp in ['train', 'dev', 'test']:
#     with open('../finetune/data/'+args.semeval+'/'+sp+'_'+args.method+'.json', 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             one = json.loads(line)
#             data_sem.append(one)
#
# from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
# tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
# model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
# model.eval()
#
# cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
#
# fea_tweet = torch.tensor([[]]).view(-1,768).cuda()
# for one in tqdm(data_tweet):
#     input = tokenizer(one['text'],truncation=True)
#     with torch.no_grad():
#         outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
#                         attention_mask=torch.tensor([input['attention_mask']]).cuda(),
#                         # token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
#                         output_hidden_states=True, return_dict=True).pooler_output
#     fea_tweet = torch.cat((fea_tweet,outputs),0)
#
# fea_sem = torch.tensor([[]]).view(-1,768).cuda()
# for one in tqdm(data_sem):
#     one = one['text'].split(' \n ')[-2]
#     input = tokenizer(one, truncation=True)
#     with torch.no_grad():
#         outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
#                         attention_mask=torch.tensor([input['attention_mask']]).cuda(),
#                         # token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
#                         output_hidden_states=True, return_dict=True).pooler_output
#     fea_sem = torch.cat((fea_sem,outputs),0)
#
# if not os.path.isdir('eval-' + task + '_evensplit' + str(epoch)):
#     os.mkdir('eval-' + task + '_evensplit' + str(epoch))
#
# for idx in range(len(fea_tweet)):
#     fea_one_tweet = fea_tweet[idx]
#     dis = cos_sim(fea_one_tweet,fea_sem)
#     val,best_idx = dis.topk(1)
#     print(data_tweet[idx])
#     print(data_sem[best_idx[0]])


print('loading model')
from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base').cuda()
model.eval()
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()

print('loading data')
data_sem = []
for sp in ['train', 'dev', 'test']:
    with open('../finetune/data/' + args.semeval + '/' + sp + '_' + args.method + '.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            one = json.loads(line)
            data_sem.append(one)

fea_sem = torch.tensor([[]]).view(-1,768).cuda()
for one in tqdm(data_sem):
    one = preprocess(one['text'].split(' \n ')[-2])
    input = tokenizer(one, truncation=True)
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
                        attention_mask=torch.tensor([input['attention_mask']]).cuda(),
                        # token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
                        output_hidden_states=True, return_dict=True).pooler_output
    fea_sem = torch.cat((fea_sem,outputs),0)

for sp in ['train','dev','test']:
    data_tweet = []
    with open('../finetune/data/'+args.tweeteval+'/'+sp+'.json','r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            one = json.loads(line)
            data_tweet.append(one)

    fea_tweet = torch.tensor([[]]).view(-1,768).cuda()
    for one in tqdm(data_tweet):
        input = tokenizer(preprocess(one['text']),truncation=True)
        with torch.no_grad():
            outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
                            attention_mask=torch.tensor([input['attention_mask']]).cuda(),
                            # token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
                            output_hidden_states=True, return_dict=True).pooler_output
        fea_tweet = torch.cat((fea_tweet,outputs),0)

    with open('../finetune/data/'+args.tweeteval+'/'+sp+'_'+args.method+'.json','w',encoding='utf-8') as f:
        for idx in trange(len(fea_tweet)):
            fea_one_tweet = fea_tweet[idx]
            dis = cos_sim(fea_one_tweet,fea_sem)
            val,best_idx = dis.topk(2)
            if (val[0] - val[1]) < 0.1:
                print(val[0])
                print(data_tweet[idx]['text'])
                print(data_sem[best_idx[0]]['text'].split(' \n ')[-2])

                print(val[1])
                print(data_tweet[idx]['text'])
                print(data_sem[best_idx[1]]['text'].split(' \n ')[-2])
            tmp = json.dumps(data_sem[best_idx[0]], ensure_ascii=False)
            f.write(tmp + '\n')