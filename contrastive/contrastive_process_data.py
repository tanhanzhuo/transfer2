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
parser.add_argument('--hash_file',default='feature_model10001000_num1000100_together',type=str)
parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
# parser.add_argument('--model',default='checkpoint-1000000',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=2, type=int)
parser.add_argument('--method',default='model10001000_num1000100',type=str)
#simcse
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

def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append({'labels': line.split('\t')[0], 'text': line.split('\t')[1]})
    return data
import json
def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

hash_samples = []
hash_embs = []
for idx in range(1):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.extend(tmp['center_samples'])
    hash_embs.extend(tmp['center_embs'])
    tmp.close()
hash_embs = torch.tensor(hash_embs).cuda()

for task in args.task_name.split(','):
    for fileName in ['train', 'dev', 'test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName)
        data_hash = copy.deepcopy(train_dataset)
        for idx in trange(len(train_dataset)):
            one = train_dataset[idx]
            input = tokenizer(one['text'])
            with torch.no_grad():
                outputs = model(input_ids=torch.tensor([input['input_ids']]).cuda(),
                                attention_mask=torch.tensor([input['attention_mask']]).cuda(),
                                token_type_ids=torch.tensor([input['token_type_ids']]).cuda(),
                                output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
                dis = -torch.linalg.vector_norm(outputs-hash_embs,dim=1).cpu()
                best_idx = np.argpartition(np.array(dis), -args.best)[-args.best:]
                for cur_idx in best_idx:
                    data_hash[idx]['text'] = emoji.emojize(tokenizer.decode(hash_samples[cur_idx][1:]).replace(tokenizer.pad_token,'').strip()) \
                                             + ' ' + data_hash[idx]['text']

        write_json(data_hash, args.dataset_path + task + '/' + fileName + args.method)

    print('task done! {}'.format(task))