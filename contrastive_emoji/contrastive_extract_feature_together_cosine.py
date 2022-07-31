#  CUDA_VISIBLE_DEVICES=0 python contrastive_extract_feature_together_cosine.py --file selected_thre100_num10000 --model /work/SimCSE-main/result/thre100_num100/ --num_sample 10 --save feature_modelT100N100_fileT100_num10 --CUR_SPLIT 0
import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
from scipy.spatial.distance import pdist, squareform
import time
# from accelerate import Accelerator
# accelerate = Accelerator()
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='selected_noemoji_thre100000_num100000.json',type=str)
parser.add_argument('--model',default='/work/SimCSE-main/results/emoji_thre100000_num50000_32/9/',type=str)
parser.add_argument("--num_sample", default=10000, type=int)
parser.add_argument('--save',default='feature_modelT100000N50000_fileT100000_num10000',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--preprocessing_num_workers", default=10, type=int)
parser.add_argument("--batch_size", default=7, type=int)

#simcse
parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

#####splits
parser.add_argument("--NUM_SPLIT", default=1, type=int)
parser.add_argument("--CUR_SPLIT", default=0, type=int)

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
text_column_name = "text"
def tokenize_function(examples):
    return tokenizer(
        examples[text_column_name],
        padding=False,
        truncation=True,
        max_length=args.max_seq_length,
        # return_special_tokens_mask=True,
    )

with open(args.file+'_index.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)

SPLIT = args.NUM_SPLIT
TOTAL = len(CONVERT.keys())
BATCH = int(TOTAL /SPLIT)
IDX = []
for idx in range(SPLIT-1):
    IDX.append([BATCH*idx, BATCH*(idx+1)])
IDX.append([BATCH*(idx+1), TOTAL])

raw_datasets = datasets.load_dataset('json', data_files=args.file+'_'+str(args.CUR_SPLIT)+'.json')
tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=args.preprocessing_num_workers,
    remove_columns=[text_column_name],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset line_by_line",
)
print('total tokenized samples')
print(len(tokenized_datasets['train']))
# tokenized_datasets['train'] = tokenized_datasets['train'].add_column('labels', list(range(len(tokenized_datasets['train']))))
batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer)
train_data_loader = DataLoader(
    tokenized_datasets['train'], shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
)
MAX_LEN=len(train_data_loader)
progress_bar = tqdm(range(BATCH))
total_num = 0
embeddings = []
tmp_samples = []
center_samples = []
center_embs = []
center_hash = []
previous_label = tokenized_datasets['train'][0]['labels']
# print(previous_label)
for step, batch in enumerate(train_data_loader):
    with torch.no_grad():
        labels= batch['labels']
        if labels[0] != labels[-1] or labels[0] != previous_label or step == MAX_LEN:#goes to another hashtag
            total_num+=1
            # print(embeddings.shape)
            # print('start calculate')
            # curr_time = time.time()
            dis = squareform(pdist(embeddings, 'cosine'))
            # print('end calculate')
            # print(time.time()-curr_time)
            # curr_time = time.time()
            dis_sum = np.sum(-dis, axis=1)#######pdist cosine vs nn.cos : 1-cos
            best = np.argpartition(np.array(dis_sum), -args.num_sample)[-args.num_sample:]
            # print('end rank')
            # print(time.time() - curr_time)
            # curr_time = time.time()
            for idx in best:
                center_embs.append(embeddings[idx])
                center_samples.append(tmp_samples[idx])
            center_hash.append(CONVERT[str(previous_label.item())])
            # print('end save')
            # print(time.time() - curr_time)
            # curr_time = time.time()
            print('current hashtag:{}, {}, number hashtag:{}, cur hash sample:{}, total hash samples:{}'. \
                  format(previous_label.item(),CONVERT[str(previous_label.item())], total_num, len(embeddings), len(center_samples)))
            with open(args.save+'_'+str(args.CUR_SPLIT)+'.txt', 'a', encoding='utf-8') as f:
                f.write('current hashtag:{}, {}, number hashtag:{}, cur hash sample:{}, total hash samples:{} \n'. \
                  format(previous_label.item(),CONVERT[str(previous_label.item())], total_num, len(embeddings), len(center_samples)))
            del embeddings, dis, dis_sum
            torch.cuda.empty_cache()
            embeddings = []
            tmp_samples = []
            # print('end restart')
            # print(time.time() - curr_time)
            # curr_time = time.time()
            progress_bar.update(1)
        else:
            tmp_samples.extend(batch['input_ids'])
            outputs = model(input_ids=batch['input_ids'].cuda(),
                            attention_mask=batch['attention_mask'].cuda(),
                            token_type_ids=batch['token_type_ids'].cuda(),
                            output_hidden_states=True, return_dict=True,sent_emb=True).pooler_output
            embeddings.extend(outputs.cpu().numpy())
        previous_label = labels[-1]

np.savez(args.save+'_'+str(args.CUR_SPLIT),center_samples=np.array(center_samples),\
         center_embs=np.array(center_embs),center_hash=np.array(center_hash))

'''
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
texts = [
    "A woman is reading.",
    "A man is playing a guitar.",
    "A kid is inside the house."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True,sent_emb=True).pooler_output

cosine(embeddings[0], embeddings[1])
cosine(embeddings[0], embeddings[2])
# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])
'''