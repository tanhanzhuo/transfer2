import argparse
import os
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig,AutoModelForSequenceClassification,DataCollatorWithPadding
import datasets
import torch
from accelerate import Accelerator
import copy
from tqdm import trange,tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--method", default='', type=str, required=False, help="method for training emoji prediction, e.g. cluster")
parser.add_argument("--num_classes", default=20, type=int, help="number of emoji classes")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
parser.add_argument("--model_name_or_path", default='/work/transfer2/emoji/model', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

# parser.add_argument("--model_name", default='/work/transfer2/emoji/model', type=str, required=False, help="tokenizer name")

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
            json.dump(one, f)
            f.write('\n')

@torch.no_grad()
def pred_prob(args):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_classes)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model = accelerator.prepare(model)
    model.eval()

    for task in args.task_name.split(','):
        for fileName in ['train', 'dev', 'test']:
            train_dataset = read_data(args.dataset_path + task + '/' + fileName)
            data_emoji = copy.deepcopy(train_dataset)
            for idx in trange(len(train_dataset)):
                one = train_dataset[idx]
                input = tokenizer(one['text'])['input_ids']
                outputs = model(torch.tensor([input]).cuda())
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                data_emoji[idx]['emoji'] = int(preds[0])
            write_json(data_emoji,args.dataset_path + task + '/' + fileName + args.method)

        print('task done! {}'.format(task))


if __name__ == "__main__":
    args = parser.parse_args()
    pred_prob(args)