import argparse
import os
import random
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import json

from modeling import RobertaForTokenClassification
import numpy as np
import paddle
from tqdm import tqdm,trange
from paddlenlp.data import Pad

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='/work/test/finetune_newdata/data/hate', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='/work/test/finetune_newdata/data/hate', type=str, required=False, help="dataset name")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
parser.add_argument("--tokenizer_name", default='vinai/bertweet-base', type=str, required=False, help="tokenizer name")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--preprocessing_num_workers", default=1, type=int, help="multi-processing number.")
parser.add_argument("--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets")

parser.add_argument("--CUR_GPU", default=1, type=int)
parser.add_argument("--model_name", default='/work/test/pretrain_hashtag/keyphrase/model/twitter_hash_key', type=str, required=False, help="tokenizer name")



def read_data(fileName):
    with open(fileName, 'r') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append({'label':line.split('\t')[0], 'text':line.split('\t')[1]})
    return data

def write_json(fileName):
    data = read_data(fileName)
    with open(fileName + '.json', 'w') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

def tokenization(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # Get the datasets: 
    if args.dataset_path is not None:
        if not os.path.isfile(args.dataset_path + '/train.json'):
            for fileName in ['train', 'dev', 'test']:
                write_json(args.dataset_path + '/' + fileName)
        data_files = {}
        data_files["train"] = args.dataset_path + '/train.json'
        data_files["dev"] = args.dataset_path + '/dev.json'
        data_files["test"] = args.dataset_path + '/test.json'
        raw_datasets = load_dataset('json', data_files=data_files)


    # Load pretrained tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, normalization=True)

    # First we tokenize all the texts.
    
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = False
    def tokenize_function(examples):
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    # tokenized_datasets.save_to_disk(args.output_dir)
    return tokenized_datasets

def pred_prob(args, tokenized_datasets):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)#, normalization=True)
    model = RobertaForTokenClassification.from_pretrained(args.model_name, num_classes=2)
    model.eval()
    USER = tokenizer('@USER',add_special_tokens=False)['input_ids'][0]
    URL = tokenizer('HTTPURL',add_special_tokens=False)['input_ids'][0]
    START = tokenizer.bos_token_id
    END = tokenizer.eos_token_id
    SP = [USER, URL, START, END]

    BS = 256
    pad1 = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32')
    pad2 = Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32')
    

    for SPLIT in ['train', 'dev', 'test']:
        train_dataset = tokenized_datasets[SPLIT]
        prob_write = []
        for idx_train in trange(int(len(train_dataset)/BS)+1):
            start_cur = idx_train*BS
            end_cur = min( (idx_train+1)*BS, len(train_dataset) )
            # if idx_train < 441:
            #     continue
            # else:
            #     aa =1
            batch = train_dataset[start_cur:end_cur]
            input_ids_batch = batch['input_ids']

            txt_small_token_join_all = []
            txt_small_token_len_all = []
            txt_token_clean_len_all = []
            prob_all = []
            txt_type_id_all = []
            for txt_token in input_ids_batch:
                txt_token = np.array(txt_token)
                # if len(txt_token) > 128 or len(txt_token) < 4:
                #     return []
                prob = np.array([-1.0] * len(txt_token))
                for idx in range(len(txt_token)):
                    if txt_token[idx] in SP:
                        prob[idx] = 0

                txt_token_clean = txt_token[prob!=0]
                txt_token_sep=[[]]
                idx_cur = 0
                for idx in range(len(txt_token_clean)):
                    # one = tokenizer.decode([txt_token_clean[idx]])
                    one = tokenizer._convert_id_to_token(txt_token_clean[idx])
                    if one[-2:] != '@@':
                        txt_token_sep[idx_cur].append(txt_token_clean[idx])
                        idx_cur+=1
                        if idx!=len(txt_token_clean)-1:
                            txt_token_sep.append([])
                    else:
                        txt_token_sep[idx_cur].append(txt_token_clean[idx])
                txt_token_clean_len = [len(token) for token in txt_token_sep]

                txt_small_token_sep=[]
                txt_small_token_join=[]
                for token in txt_token_sep:
                    word = tokenizer.decode(token)
                    word_small = word.lower()

                    word_small_token = tokenizer._convert_token_to_id(word_small)
                    if word_small_token == tokenizer.unk_token_id:
                        word_small_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_small))#tokenizer(word_small,add_special_tokens=False)['input_ids']
                    else:
                        word_small_token = [word_small_token]
                    
                    txt_small_token_sep.append(word_small_token)
                    txt_small_token_join.extend(word_small_token)
                txt_small_token_len = [len(token) for token in txt_small_token_sep]
                txt_small_token_join=[START] + txt_small_token_join + [END]

                txt_small_token_join_all.append(txt_small_token_join)
                txt_small_token_len_all.append(txt_small_token_len)
                txt_token_clean_len_all.append(txt_token_clean_len)
                prob_all.append(prob)
                txt_type_id_all.append( [0]*len(txt_small_token_join) )
            
            with paddle.no_grad():
                logits = model(paddle.to_tensor(pad1(txt_small_token_join_all)), paddle.to_tensor(pad2(txt_type_id_all)))
                preds_all=list(paddle.nn.functional.softmax(logits)[:,1:-1,1].cpu().numpy())
                # preds = [0]*(len(txt_small_token_join)-2)
            # preds_all = pad2(txt_type_id_all)
            for idx_bs in range(len(input_ids_batch)):
                txt_small_token_join=txt_small_token_join_all[idx_bs]
                txt_small_token_len=txt_small_token_len_all[idx_bs]
                txt_token_clean_len=txt_token_clean_len_all[idx_bs]
                prob=prob_all[idx_bs]
                if np.sum(np.abs(prob)) != 0:
                    preds = list(preds_all[idx_bs])
                    preds_reshape = []
                    for idx in range(len(txt_small_token_len)):
                        pop_len = txt_small_token_len[idx]
                        preds_reshape.append([])
                        for idx2 in range(pop_len):
                            preds_reshape[idx].append(preds.pop(0))
                    try:
                        preds_reshape_max = [max(tmp) for tmp in preds_reshape]
                    except:
                        a=1

                    prob_tmp =[]
                    for idx in range(len(txt_token_clean_len)):
                        token_len = txt_token_clean_len[idx]
                        for idx2 in range(token_len):
                            prob_tmp.append(preds_reshape_max[idx])

                    idx_tmp = 0
                    for idx in range(len(prob)):
                        if prob[idx] == 0:
                            continue
                        prob[idx] = prob_tmp[idx_tmp]
                        idx_tmp+=1
                prob_write.append(prob)
        train_dataset = train_dataset.add_column("prob_map", prob_write)
        tokenized_datasets[SPLIT] = train_dataset
    return tokenized_datasets

if __name__ == "__main__":
    args = parser.parse_args()
    paddle.set_device("gpu:"+str(args.CUR_GPU))
    tokenized_datasets = tokenization(args)
    tokenized_datasets = pred_prob(args, tokenized_datasets)
    tokenized_datasets.save_to_disk(args.output_dir+'/prob')