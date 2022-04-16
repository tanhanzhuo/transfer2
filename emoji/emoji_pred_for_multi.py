import argparse
import os
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig,AutoModelForSequenceClassification,DataCollatorWithPadding
import datasets
import torch
import accelerate

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-17,sem-18,wtwt', type=str, required=False, help="dataset name")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
parser.add_argument("--model_name_or_path", default='vinai/bertweet-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

# parser.add_argument("--model_name", default='/work/transfer2/emoji/model', type=str, required=False, help="tokenizer name")
def convert_example(example, label2idx):
    example.pop('special_tokens_mask')
    example.pop('attention_mask')
    example['label'] = label2idx[example['label']]
    example['prob_map'] = [int(0) if i < 0.5 else int(1) for i in example['prob_map']]

    return example  # ['input_ids'], example['token_type_ids'], label, prob

@torch.no_grad()
def pred_prob(args):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=20)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model = accelerator.prepare(model)
    model.eval()

    for task in args.task_name.split(','):
        tokenized_datasets = datasets.load_from_disk(args.dataset_path+task+'/prob')
        for SPLIT in ['train', 'dev', 'test']:
            train_dataset = tokenized_datasets[SPLIT]
            labels = train_dataset['label']
            train_dataset = train_dataset.remove_columns(['prob_map', 'label', 'special_tokens_mask'])
            batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer)
            train_data_loader = DataLoader(
                train_dataset, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
            )
            train_data_loader = accelerator.prepare(train_data_loader)
            emoji_pred = []
            for step, batch in enumerate(train_data_loader):
                # batch.pop('special_tokens_mask')
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                emoji_pred.extend(preds)
            train_dataset = train_dataset.add_column("emoji", emoji_pred)
            train_dataset = train_dataset.add_column("labels", labels)
            tokenized_datasets[SPLIT] = train_dataset
        tokenized_datasets.save_to_disk(args.output_dir + task + '/emoji')


if __name__ == "__main__":
    args = parser.parse_args()
    pred_prob(args)