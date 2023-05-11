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
parser.add_argument('--hash_file',default='./tweet_hash_clean_seg_one20/tweet_hash_clean_seg',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
# parser.add_argument('--model_name',default='/work/SimCSE-main/result/thre100_num100_remove/1399999',type=str)
parser.add_argument('--model_name',default='../lmbff/contrastive_models/one/20_new/',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=100, type=int)
parser.add_argument('--method',default='_seg_one20',type=str)
parser.add_argument("--split", default=100, type=int)#for gpu memory
parser.add_argument("--hashprocess", default='seg', type=str)#for gpu memory

args = parser.parse_args()

with open('../contrastive/hash_seg10.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    line = line.strip()
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]

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
            # data.append({'labels': line.split('\t')[0], 'text': line.split('\t')[1]})
            data.append(json.loads(line))
    return data

import re
import string
HASH = re.compile(r"#\S+")
def process(line):
    hash_tmp = HASH.findall(line)
    hash_tmp_clean = []
    for hash_one in hash_tmp:
        # hash_one = hash_one.lower()
        if len(hash_one) > 30:
            continue
        if hash_one[1].isalpha():
            if hash_one[-1] == '…':
                continue
            if len(hash_one) > 3 and hash_one[-3:] == '...':
                continue
            if hash_one[-1] in string.punctuation:
                hash_one = hash_one[:-1]
            hash_clean = re.findall('[a-zA-Z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                hash_tmp_clean.append(hash_one)

    return hash_tmp_clean

def read_data_hashremove(fileName, args):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            # data_tmp = line.split('\t')[1].strip()
            data_tmp = json.loads(line)['text']
            hash_tmp = process(data_tmp)

            if args.hashprocess == 'same':
                data_tmp = data_tmp
            elif args.hashprocess == 'remove':
                for hash_two in hash_tmp:
                    data_tmp = data_tmp.replace(hash_two, '')
            elif args.hashprocess == 'seg':
                for hash_two in hash_tmp:
                    tmp2 = hash_seg.get(hash_two.lower())
                    if tmp2 is not None:
                        data_tmp = data_tmp.replace(hash_two, tmp2)
                    else:
                        data_tmp = data_tmp.replace(hash_two, hash_two[1:])

            data.append({'labels': json.loads(line)['labels'], 'text':data_tmp })
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
    # if idx < args.split/2:
    #     hash_embs.append(torch.tensor(tmp['embs'],dtype=torch.float16).cuda(0))
    # else:
    #     hash_embs.append(torch.tensor(tmp['embs'], dtype=torch.float16).cuda(1))
    hash_embs.append(torch.tensor(tmp['embs'],dtype=torch.float16))###############save gpu memory

    tmp.close()

# hash_embs= torch.tensor(np.array(hash_embs))
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda(0)
for task in args.task_name.split(','):
    for fileName in ['train', 'dev', 'test']:
    # for fileName in ['test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName + '.json')
        data_hash_all = copy.deepcopy(train_dataset)
        train_dataset = read_data_hashremove(args.dataset_path + task + '/' + fileName + '.json')
        for idx in trange(len(train_dataset)):
            one = train_dataset[idx]
            input = tokenizer(one['text'],truncation=True)
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(0),
                                attention_mask=torch.tensor([input['attention_mask']]).cuda(0),
                                token_type_ids=torch.tensor([input['token_type_ids']]).cuda(0),
                                output_hidden_states=True, return_dict=True).pooler_output
                best_distance = []
                best_text = []
                for sp in range(args.split):
                    # if sp < args.split/2:
                    #     outputs = torch.tensor(outputs, dtype=torch.float16).cuda(0)
                    # else:
                    #     outputs = torch.tensor(outputs, dtype=torch.float16).cuda(1)
                    # dis = cos_sim(outputs, hash_embs[sp])
                    outputs = torch.tensor(outputs, dtype=torch.float16).cuda()
                    dis = cos_sim(outputs,hash_embs[sp].cuda())

                    val,best_idx = dis.topk(args.best)
                    for tmp_idx in best_idx.cpu().numpy():
                        best_distance.append(dis[tmp_idx].cpu().numpy())
                        best_text.append(hash_samples[sp][tmp_idx])
                        # best_word.append(hash_word[hash_tags[sp][tmp_idx]])#list of keywords
                    del dis
                    torch.cuda.empty_cache()

                best_idx = np.flip( np.argsort(np.array(best_distance))[-args.best:] )
                for cur_idx in range(0,len(best_idx)):
                    data_hash_all[idx]['text'+str(cur_idx)] = best_text[best_idx[cur_idx]].strip()

        write_json(data_hash_all, args.dataset_path + task + '/' + fileName + args.method + '_top' + str(args.best) + '_sp')

    print('task done! {}'.format(task))