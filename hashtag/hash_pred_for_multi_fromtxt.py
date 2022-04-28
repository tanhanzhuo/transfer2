import argparse
import os
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig,AutoModelForSequenceClassification,DataCollatorWithPadding
import datasets
import torch
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default=None, type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance/face_masks_new,stance/fauci_new,stance/school_closures_new,stance/stay_at_home_orders_new,hate,sem-17,sem-18,wtwt/CVS_AET,wtwt/CI_ESRX,wtwt/ANTM_CI,wtwt/AET_HUM,wtwt/FOXA_DIS', type=str, required=False, help="dataset name")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
parser.add_argument("--model_name_or_path", default='/work/transfer2/hashtag/model_100_1000_epoch20', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")
parser.add_argument("--num", default=100, type=int, help="top hashtags")

# parser.add_argument("--model_name", default='/work/transfer2/emoji/model', type=str, required=False, help="tokenizer name")

@torch.no_grad()
def pred_prob(args):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model = accelerator.prepare(model)
    model.eval()

    for task in args.task_name.split(','):
        tokenized_datasets = datasets.load_from_disk(args.dataset_path+task+'/token')
        for SPLIT in ['train', 'dev', 'test']:
            train_dataset = tokenized_datasets[SPLIT]
            labels = train_dataset['label']
            train_dataset = train_dataset.remove_columns(['label', 'special_tokens_mask'])
            batchify_fn = DataCollatorWithPadding(tokenizer=tokenizer)
            train_data_loader = DataLoader(
                train_dataset, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
            )
            train_data_loader = accelerator.prepare(train_data_loader)
            hash_pred = []
            for step, batch in enumerate(train_data_loader):
                # batch.pop('special_tokens_mask')
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                hash_pred.extend(preds)
            train_dataset = train_dataset.add_column("hash", hash_pred)
            train_dataset = train_dataset.add_column("labels", labels)
            tokenized_datasets[SPLIT] = train_dataset
        if args.output_dir is None:
            tokenized_datasets.save_to_disk(args.dataset_path + task + '/hash_' + str(args.num))
        else:
            tokenized_datasets.save_to_disk(args.output_dir + task + '/hash_' + str(args.num))
        print('task done! {}'.format(task))


if __name__ == "__main__":
    args = parser.parse_args()
    pred_prob(args)