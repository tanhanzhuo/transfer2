import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer,DataCollatorWithPadding
import datasets
import argparse
from tqdm import tqdm
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default='twitter_hash', type=str, required=False, help="dataset name")
parser.add_argument("--model_name", default='princeton-nlp/sup-simcse-roberta-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=64, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--save", default='./features/', type=str, required=False, help="dataset name")
parser.add_argument("--split", default=200, type=int, required=False, help="dataset name")

args = parser.parse_args()

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name).cuda()
model.eval()
datafull = datasets.load_from_disk(args.dataset_path)
datafull = datafull['train']
batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer,max_length=args.max_seq_length)
data_loader = DataLoader(
    datafull, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
)
BATCH = int(len(datafull) / args.split)
samples = []
embs= []
BATCH_IDX=0
progress_bar = tqdm(range(len(data_loader)))
if not os.path.isdir(args.save):
    os.mkdir(args.save)
with torch.no_grad():
    for step, batch in enumerate(data_loader):
        embeddings = model(input_ids=batch['input_ids'].cuda(),
                           attention_mask=batch['attention_mask'].cuda(),
                           output_hidden_states=True, return_dict=True).pooler_output
        samples.extend(batch['input_ids'].numpy())
        embs.extend(embeddings.cpu().numpy())
        progress_bar.update(1)
        if len(samples) >= BATCH:
            np.savez(args.save+args.dataset_path+'_'+str(BATCH_IDX),samples=np.array(samples),embs=np.array(embs))
            samples = []
            embs = []
            BATCH_IDX += 1

# Tokenize input texts
# texts = [
#     "There's a kid on a skateboard.",
#     "A kid is skateboarding.",
#     "A kid is inside the house."
# ]
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
# cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
# # Get the embeddings
# with torch.no_grad():
#     embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
