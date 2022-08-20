import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer,AutoConfig,DataCollatorWithPadding
import datasets
import argparse
from tqdm import tqdm
import os
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default='twitter_hash_join_thre100_num1000_hashremove', type=str, required=False, help="dataset name")
parser.add_argument("--model_name", default='/work/SimCSE-main/results/thre100_num100_hashremove/13999', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=64, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--save", default='./features_hashremove/', type=str, required=False, help="dataset name")
parser.add_argument("--split", default=100, type=int, required=False, help="dataset name")

parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

args = parser.parse_args()

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class MyDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = []
        for one in features:
            text.append(one.pop('text'))
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        batch['text'] = text
        return batch


# Import our models. The package will take care of downloading the models automatically
from models import RobertaForCL
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
config = AutoConfig.from_pretrained(args.model_name)
model = RobertaForCL.from_pretrained(args.model_name,config=config,model_args=args).cuda()
model.eval()
datafull = datasets.load_from_disk(args.dataset_path)
datafull = datafull['train']
batchify_fn = MyDataCollatorWithPadding(tokenizer=tokenizer,max_length=args.max_seq_length)
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
                           token_type_ids=batch['token_type_ids'].cuda(),
                           output_hidden_states=True, return_dict=True,sent_emb=True).pooler_output
        samples.extend(batch['text'])
        embs.extend(embeddings.cpu().numpy())
        progress_bar.update(1)
        if len(samples) >= BATCH or step == len(data_loader)-1:
            np.savez(args.save+args.dataset_path+'_'+str(BATCH_IDX),embs=np.array(embs),samples=np.array(samples))
            # with open(args.save+args.dataset_path+'_'+str(BATCH_IDX)+'.txt', 'w',encoding='utf-8') as f:
            #     for line in samples:
            #         f.write(line+ ' \n')
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
