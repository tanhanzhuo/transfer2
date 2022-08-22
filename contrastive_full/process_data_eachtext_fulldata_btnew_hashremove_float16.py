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
# parser.add_argument('--model_name',default='/work/SimCSE-main/result/thre100_num100_remove/1399999',type=str)
parser.add_argument('--model_name',default='./1399999_new/',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=20, type=int)
parser.add_argument('--method',default='_fulldata_bt_hashremove',type=str)
parser.add_argument("--split", default=50, type=int)#for gpu memory

args = parser.parse_args()

from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
config = AutoConfig.from_pretrained(args.model_name)
# model = AutoModel.from_pretrained(args.model,config=config).cuda()
model = AutoModel.from_pretrained(args.model_name).cuda(0)
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

hash_samples = []
hash_embs = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.append(tmp['samples'])
    # hash_embs.extend(tmp['center_embs'])
    if idx < args.split/2:
        hash_embs.append(torch.tensor(tmp['embs'],dtype=torch.float16).cuda(0))
    else:
        hash_embs.append(torch.tensor(tmp['embs'], dtype=torch.float16).cuda(1))
    tmp.close()

# hash_embs= torch.tensor(np.array(hash_embs))
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda(0)
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
                outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(0),
                                attention_mask=torch.tensor([input['attention_mask']]).cuda(0),
                                token_type_ids=torch.tensor([input['token_type_ids']]).cuda(0),
                                output_hidden_states=True, return_dict=True).pooler_output
                # dis = -np.linalg.norm(outputs.cpu().numpy()-hash_embs,axis=1)
                best_distance = []
                best_text = []
                for sp in range(args.split):
                    if sp < args.split/2:
                        outputs = torch.tensor(outputs, dtype=torch.float16).cuda(0)
                    else:
                        outputs = torch.tensor(outputs, dtype=torch.float16).cuda(1)
                    dis = cos_sim(outputs,hash_embs[sp])
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