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
parser.add_argument('--hash_file',default='./features_hashremove/twitter_hash_join_thre100_num100_hashremove',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
parser.add_argument('--model_name',default='/work/SimCSE-main/result/thre100_num100_remove/1399999',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=20, type=int)
parser.add_argument('--method',default='_fulldata_bt_hashremove',type=str)
parser.add_argument("--split", default=50, type=int)#for gpu memory
parser.add_argument("--comb", default=5, type=int)#for gpu memory

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
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
config = AutoConfig.from_pretrained(args.model_name)
# model = AutoModel.from_pretrained(args.model,config=config).cuda()
model = RobertaForCL.from_pretrained(
                args.model_name,
                config=config,
                model_args=args
            ).cuda()
model.eval()

def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append({'labels': line.split('\t')[0], 'text': line.split('\t')[1]})
    return data

import re
HASH = re.compile(r"#\S+")
def read_data_hashremove(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            one = line.split('\t')[1]
            hash_tmp = HASH.findall(one)
            for hash_two in hash_tmp:
                one = one.replace(hash_two, '')

            data.append({'labels': line.split('\t')[0], 'text':one })
    return data

import json
def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

#########comb
hash_samples = []
hash_embs = []
if args.split % args.comb != 0:
    print('error!!!')
    raise ValueError('comb must be divided by splits')
args.split = int(args.split/args.comb)
for idx in trange(args.split):
    hash_samples_tmp = []
    hash_embs_tmp = []
    for idx2 in range(args.comb):
        tmp = np.load(args.hash_file + '_' + str(idx*args.comb+idx2) + '.npz', allow_pickle=True)
        hash_samples_tmp.extend(tmp['samples'])
        # hash_embs.extend(tmp['center_embs'])
        hash_embs_tmp.extend(tmp['embs'])
        tmp.close()
    hash_samples.append(hash_samples_tmp)
    # hash_embs.extend(tmp['center_embs'])
    hash_embs.append(torch.tensor(hash_embs_tmp))
hash_samples_tmp = []
hash_embs_tmp = []
# hash_embs= torch.tensor(np.array(hash_embs))
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
for task in args.task_name.split(','):
    for fileName in ['train', 'dev', 'test']:
    # for fileName in ['test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName)
        data_hash_all = copy.deepcopy(train_dataset)
        train_dataset = read_data_hashremove(args.dataset_path + task + '/' + fileName)
        for idx in trange(len(train_dataset)):
            one = train_dataset[idx]
            input = tokenizer(one['text'],truncation=True)
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
                                attention_mask=torch.tensor([input['attention_mask']]).cuda(),
                                token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
                                output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
                # dis = -np.linalg.norm(outputs.cpu().numpy()-hash_embs,axis=1)
                best_distance = []
                best_text = []
                for sp in range(args.split):
                    # dis = torch.linalg.vector_norm(outputs.cuda(sp) - hash_embs[sp], dim=1).cpu()
                    dis = cos_sim(outputs,hash_embs[sp].cuda())
                    # dis = dis.view(-1,args.num_samples).sum(dim=-1)##################################hash each
                    # best_idx = np.argpartition(np.array(dis), -args.best)[-args.best:]
                    val,best_idx = dis.topk(args.best)
                    for tmp_idx in best_idx.cpu().numpy():
                        best_distance.append(dis[tmp_idx].cpu().numpy())
                        best_text.append(hash_samples[sp][tmp_idx])
                        # best_word.append(hash_word[hash_tags[sp][tmp_idx]])#list of keywords
                    del dis
                    torch.cuda.empty_cache()

                best_idx = np.argsort(np.array(best_distance))[-args.best:]
                for cur_idx in best_idx:
                    data_hash_all[idx]['text'] = ' ' +best_text[cur_idx].strip()\
                                            + ' \n ' + data_hash_all[idx]['text'].strip() + ' \n '

        write_json(data_hash_all, args.dataset_path + task + '/' + fileName + args.method + '_top' + str(args.best)\
                   +'_'+'textfirst')

    print('task done! {}'.format(task))