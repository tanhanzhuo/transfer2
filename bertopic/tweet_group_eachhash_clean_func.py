import argparse
import json
import random
import copy
import torch
from tqdm import tqdm,trange
import numpy as np
import re
import string
from bertopic import BERTopic
from transformers.pipelines import pipeline
from sentence_transformers import SentenceTransformer
# from memory_profiler import profile

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='../pretrain/hashtag/tweet_hash_clean_group_all.txt',type=str)
parser.add_argument('--hash_list',default='hash_thre_list.txt',type=str)
parser.add_argument('--num',default=100,type=int)
parser.add_argument('--name',default='tweet_hash_clean_group_subgroup',type=str)
parser.add_argument('--split',default=4,type=int)
parser.add_argument('--split_cur',default=0,type=int)

args = parser.parse_args()

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

def write_json(fileName, data):
    with open(fileName + '.json', 'a', encoding='utf-8') as f:
        for one in tqdm(data):
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')

def read_data(args):
    hash_thre_list = []
    with open(args.hash_list, 'r', encoding='utf-8') as f:
        for line in f:
            hash_thre_list.append(line.strip())

    split_num = int(len(hash_thre_list)/args.split)
    split_s = split_num * args.split_cur
    if args.split_cur == args.split -1:
        split_e = len(hash_thre_list)
    else:
        split_e = split_num * (args.split_cur + 1)
    hash_thre_list_split = hash_thre_list[split_s:split_e]

    hash_thre_list_split_dic = {}
    for hash_one in hash_thre_list_split:
        hash_thre_list_split_dic[hash_one] = 0
    hash_data = {}
    with open(args.file, 'r', encoding='utf-8') as f:
        cur_hash = ''
        # lines = f.readlines()
        for line in tqdm(f):
            if line[:10] == 'TANS_HASH:':
                cur_hash = line.strip().split(':')[-1]
                tmp = hash_thre_list_split_dic.get(cur_hash, None)
                if tmp == 0:
                    hash_data[cur_hash] = []
                else:
                    cur_hash = ''
                continue
            if cur_hash != '':
                hash_data[cur_hash].append(line.strip())
    return hash_data, hash_thre_list_split

def group_one(hash_data_one, hash_one):
    # embedding_model = pipeline("feature-extraction", model="princeton-nlp/sup-simcse-roberta-base", device=0)
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda')
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
    topic_model = BERTopic(embedding_model=embedding_model, verbose=False)

    hash_data_two = []
    for data_tmp in hash_data_one:
        data_tmp = data_tmp.replace('@USER', '').replace('https', '')
        hash_tmp = process(data_tmp)
        for hash_two in hash_tmp:
            if hash_two.lower() == hash_one:
                data_tmp = data_tmp.replace(hash_two, '')
        hash_data_two.append(data_tmp)
    # print(hash_data_two)
    topics, probs = topic_model.fit_transform(hash_data_two)
    num_topic = max(topics) + 2
        # hash_data_one_group = {'hashtag': hash_one, 'text':[[]]*num_topic, 'emb':[]}
    # for idx in range(len(hash_data_one)):
    #     if topics[idx] + 1 in hash_data_one_group.keys():
    #         hash_data_one_group[topics[idx] + 1].append(hash_data_one[idx])
    #     else:
    #         hash_data_one_group[topics[idx] + 1] = [hash_data_one[idx]]
    text_list = []
    for i in range(num_topic):
        text_list.append([])
    hash_data_one_group = {'hashtag': hash_one, 'text':text_list, 'emb':[]}
    for idx in range(len(hash_data_one)):
        hash_data_one_group['text'][topics[idx] + 1].append(hash_data_one[idx])
    if num_topic == len(topic_model.topic_embeddings_):
        for idx in range(num_topic):
            hash_data_one_group['emb'].append(list(topic_model.topic_embeddings_[idx]))
    return hash_data_one_group

    # del embedding_model, topic_model

def main(args, hash_data, hash_thre_list_split):
    hash_data_group = []
    for hash_one in tqdm(hash_thre_list_split):
        hash_data_one = hash_data[hash_one]
        random.shuffle(hash_data_one)
        hash_data_one_group = 'bad'
        while hash_data_one_group == 'bad':
            hash_data_one_group = group_one(hash_data_one, hash_one)
        # embedding_model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
        # topic_model = BERTopic(embedding_model=embedding_model, verbose=False)
        hash_data_group.append(hash_data_one_group)
        if len(hash_data_group) > 1000:
            write_json(args.name + '_' + str(args.num) + '_' + str(args.split_cur), hash_data_group)
            del hash_data_group,hash_data_one_group,hash_data_one
            hash_data_group = []

    write_json(args.name + '_' + str(args.num) + '_' + str(args.split_cur), hash_data_group)

if __name__ == '__main__':
    hash_data, hash_thre_list_split = read_data(args)
    main(args, hash_data, hash_thre_list_split)
