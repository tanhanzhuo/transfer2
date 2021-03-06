import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
from scipy.spatial.distance import pdist, squareform
# from accelerate import Accelerator
# accelerate = Accelerator()
parser = argparse.ArgumentParser()
parser.add_argument('--file',default='1000',type=str)
parser.add_argument('--model',default='checkpoint-1100000',type=str)
parser.add_argument("--num_sample", default=100, type=int)
parser.add_argument('--save',default=None,type=str)
parser.add_argument("--max_seq_length", default=128, type=int)
parser.add_argument("--preprocessing_num_workers", default=1, type=int)
parser.add_argument("--batch_size", default=32, type=int)

#simcse
parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

#####splits
parser.add_argument("--NUM_SPLIT", default=4, type=int)
parser.add_argument("--CUR_SPLIT", default=0, type=int)

args = parser.parse_args()

# data = []
# with open(args.file, 'r', encoding='utf-8') as f:
#     if args.file.split('.')[-1] == 'txt':
#         data = f.readlines()
#     elif args.file.split('.')[-1] == 'json':
#         for line in f:
#             tmp = json.loads(line)
#             data.append(tmp['text1'])
#             data.append(tmp['text2'])
#     else:
#         print('error!!!!!!!!!!!!!!!')

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

with open(args.file+'/index_to_hashtag.json', 'r', encoding='utf-8') as f:
    CONVERT = json.load(f)

SPLIT = args.NUM_SPLIT
TOTAL = len(CONVERT.keys())
BATCH = int(TOTAL /SPLIT)
IDX = []
for idx in range(SPLIT-1):
    IDX.append([BATCH*idx, BATCH*(idx+1)])
IDX.append([BATCH*(idx+1), TOTAL])

center_samples = []
center_embs = []

progress_bar = tqdm(range(BATCH))
for index_one in list(CONVERT.keys())[IDX[args.CUR_SPLIT][0]:IDX[args.CUR_SPLIT][1]]:
    raw_datasets = datasets.load_dataset('text', data_files=args.file+'/'+str(int(index_one)-1)+'.txt')
    if len(raw_datasets['train']) < args.num_sample*50:
        continue
    raw_datasets["train"] = raw_datasets["train"].shuffle()
    raw_datasets["train"] = raw_datasets["train"].select(range(min(args.num_sample*30,len( raw_datasets["train"]))))
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line_by_line",
    )
    # tokenized_datasets['train'] = tokenized_datasets['train'].add_column('labels', list(range(len(tokenized_datasets['train']))))
    batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    train_data_loader = DataLoader(
        tokenized_datasets['train'], shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
    )

    if len(raw_datasets['train']) > 50000:
        embeddings = []
        for step, batch in enumerate(train_data_loader):
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                                token_type_ids=batch['token_type_ids'].cuda(),
                                output_hidden_states=True, return_dict=True,sent_emb=True).pooler_output
                embeddings.extend(outputs.cpu().numpy())
        dis = squareform(pdist(embeddings))
        dis_sum  = -np.sum(dis, axis=1)
        best = np.argpartition(np.array(dis_sum), -args.num_sample)[-args.num_sample:]
        center_samples.extend([tokenized_datasets['train']['input_ids'][idx] for idx in best])
        center_embs.extend([embeddings[idx] for idx in best])
    else:
        embeddings = torch.tensor([[]]).view(-1,768).cuda()
        for step, batch in enumerate(train_data_loader):
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                                token_type_ids=batch['token_type_ids'].cuda(),
                                output_hidden_states=True, return_dict=True,sent_emb=True).pooler_output
            embeddings = torch.cat((embeddings,outputs),0)
        dis = squareform(torch.nn.functional.pdist(embeddings, p=2).cpu())
        dis_sum = -np.sum(dis, axis=1)
        best = np.argpartition(np.array(dis_sum), -args.num_sample)[-args.num_sample:]
        center_samples.extend([tokenized_datasets['train']['input_ids'][idx] for idx in best])
        center_embs.extend([embeddings[idx].cpu().numpy() for idx in best])
    del embeddings,dis,dis_sum
    torch.cuda.empty_cache()
    progress_bar.update(1)
np.savez(args.save+'_'+str(args.CUR_SPLIT),center_samples=center_samples,center_embs=center_embs)

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