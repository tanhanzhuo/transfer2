import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import time
import copy
# from accelerate import Accelerator
# accelerate = Accelerator()
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='feature_simcse_fileT100N100R_num10_cluster',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
parser.add_argument('--model',default='princeton-nlp/sup-simcse-roberta-base',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,sem-18,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm,sem18-task1-affect,sem21-task7-humor', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=20, type=int)
parser.add_argument("--num_samples", default=10, type=int)
parser.add_argument("--word", default=False, type=bool)
parser.add_argument('--method',default='_simcse_fileT100N100R_num10_cluster',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory
#simcse

args = parser.parse_args()

with open('../contrastive/hash_seg.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
from models import RobertaForCL
from torch.utils.data import DataLoader
# model = AutoModel.from_pretrained(args.model,config=config).cuda()
model = AutoModel.from_pretrained(args.model).cuda()
model.eval()
# model = accelerate.prepare(model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

import json
# f=open('./selected_thre100_num500_index.json','r',encoding='utf-8')
# hash_dic = json.load(f)
# f.close()
# f=open('./selected_thre100_num3000_word_nltk.json','r',encoding='utf-8')
# hash_word = json.load(f)
# f.close()

def read_data(fileName):
    with open(fileName+'.json', 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data
import re
HASH = re.compile(r"#\S+")
def read_data_hashseg(fileName):
    with open(fileName+'.json', 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            one_dic = json.loads(line)
            one = one_dic['text']
            hash_tmp = HASH.findall(one)
            # for hash_two in hash_tmp:
            #     one = one.replace(hash_two, '')
            for hash_two in hash_tmp:
                tmp2 = hash_seg.get(hash_two.lower())
                if tmp2 is not None:
                    one = one.replace(hash_two, tmp2)
                else:
                    one = one.replace(hash_two, hash_two[1:])
            data.append({'labels': one_dic['labels'], 'text':one })
    return data

def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

hash_samples = []
hash_embs = []
hash_tags = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.append(tmp['center_samples'])
    # hash_embs.extend(tmp['center_embs'])
    # hash_embs.append(torch.tensor(tmp['center_embs']))
    hash_embs.append(torch.tensor(tmp['center_embs'], dtype=torch.float16).cuda())
    hash_tags.append(tmp['center_hash'])
    tmp.close()

# hash_embs= torch.tensor(np.array(hash_embs))
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
for task in args.task_name.split(','):
    for fileName in ['train', 'dev', 'test']:
    # for fileName in ['test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName)
        data_hash_all = copy.deepcopy(train_dataset)
        train_dataset = read_data_hashseg(args.dataset_path + task + '/' + fileName) ###remove hash to retrieve
        for idx in trange(len(train_dataset)):
            one = train_dataset[idx]
            input = tokenizer(one['text'],truncation=True)
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
                          attention_mask=torch.tensor([input['attention_mask']]).cuda(),
                          output_hidden_states=True, return_dict=True).pooler_output
                # dis = -np.linalg.norm(outputs.cpu().numpy()-hash_embs,axis=1)
                best_distance = []
                best_text = []
                best_hash = []
                best_word = []
                for sp in range(args.split):
                    # dis = torch.linalg.vector_norm(outputs.cuda(sp) - hash_embs[sp], dim=1).cpu()
                    dis = cos_sim(outputs,hash_embs[sp])
                    # dis = dis.view(-1,args.num_samples).sum(dim=-1)##################################hash each
                    # best_idx = np.argpartition(np.array(dis), -args.best)[-args.best:]
                    val,best_idx = dis.topk(args.best)
                    for tmp_idx in best_idx.cpu().numpy():
                        best_distance.append(dis[tmp_idx].cpu().numpy())
                        best_text.append(hash_samples[sp][tmp_idx])
                        best_hash.append(hash_tags[sp][int(tmp_idx/args.num_samples)])
                        # best_word.append(hash_word[hash_tags[sp][tmp_idx]])#list of keywords
                    del dis
                    torch.cuda.empty_cache()
                best_idx = np.argsort(np.array(best_distance))[-args.best:]
                for cur_idx in best_idx:
                    data_hash_all[idx]['text'] = ' ' + best_text[cur_idx].strip() \
                                                 + ' \n ' + data_hash_all[idx]['text'].strip() + ' \n '
        write_json(data_hash_all, args.dataset_path + task + '/' + fileName + args.method + '_top' + str(args.best) \
               + '_' + 'textfirst')

    print('task done! {}'.format(task))