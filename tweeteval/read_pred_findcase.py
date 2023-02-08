import json
import matplotlib.pyplot as plt
import datasets
import numpy as np
import argparse
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

with open('pairs_hash_fuldata_bt_hashseg.pickle', 'rb') as handle:
    pairs_hashseg = pickle.load(handle)

with open('pairs_hash_fulldata_simcse.pickle', 'rb') as handle:
    pairs_simcse = pickle.load(handle)
# with open('pairs_hash.pickle', 'rb') as handle:
#     pairs_hash = pickle.load(handle)
#
# f=open('../contrastive/hash_his.json','r',encoding='utf-8')
# hash_dic = json.load(f)
# f.close()
# for hash_one in list(hash_dic.keys()):
#     if hash_dic[hash_one] < 100:
#         hash_dic.pop(hash_one)

parser = argparse.ArgumentParser()
parser.add_argument("--ori", default='pred_clean2.txt', type=str, required=False)
parser.add_argument("--ret", default='pred_fullbt_seg2.txt', type=str, required=False)
parser.add_argument("--sim", default='pred_fullsimcse2.txt', type=str, required=False)
args = parser.parse_args()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')

# tasks = 'eval-stance_clean_811_8,eval-emotion_clean_811_5,eval-irony_clean_811_3,eval-offensive_clean_811_0,eval-hate_clean_811_3,sem21-task7-humor_clean_811_2,sem22-task6-sarcasm_clean_811_2,stance_clean_811_0'.split(',')
tasks = 'eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean,stance_clean'.split(',')
preds = {}
for task in tasks:
    preds[task] = []
with open(args.ori, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    cur_task = ''
    for line in lines:
        line = line.strip()
        if line[0].isalpha():
            cur_task = line
        else:
            preds[cur_task].append([int(i) for i in line])

preds_re = {}
for task in tasks:
    preds_re[task] = []
with open(args.ret, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    cur_task = ''
    for line in lines:
        line = line.strip()
        if line[0].isalpha():
            cur_task = line
        else:
            tmp = [int(i) for i in line]
            tmp_reindex = []
            for idx in range(len(tmp)):
                tmp_reindex.append(int(tmp[pairs_hashseg[cur_task][idx]]))
            preds_re[cur_task].append(tmp_reindex)

preds_si = {}
for task in tasks:
    preds_si[task] = []
with open(args.ret, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    cur_task = ''
    for line in lines:
        line = line.strip()
        if line[0].isalpha():
            cur_task = line
        else:
            tmp = [int(i) for i in line]
            tmp_reindex = []
            for idx in range(len(tmp)):
                tmp_reindex.append(int(tmp[pairs_simcse[cur_task][idx]]))
            preds_si[cur_task].append(tmp_reindex)

CONV = {'NONE':0,'FAVOR':1,'AGAINST':2}
len_thre = list(range(10,100,10))#[5,10,15,20,25,30,35,40,45,50,100]
len_result = []
for idx in range(len(len_thre)):
    len_result.append([[],[]])

num_thre = list(range(100,100000,100))
num_result = []
for idx in range(len(num_thre)):
    num_result.append([[],[]])

for idx in range(len(tasks)-1):
    # if tasks[idx] == 'eval-emotion_clean_811_5' or tasks[idx] == 'eval-offensive_clean_811_0':
    #     continue
    if 'humor' not in tasks[idx] and 'sarcasm' not in tasks[idx] and 'irony' not in tasks[idx]:
        continue
    data_ori = datasets.load_from_disk('../finetune/data/'+tasks[idx]+'/token')
    data_ori_test = data_ori['test']
    data_ori_test2 = []
    for one in data_ori_test:
        data_ori_test2.append(one)
    data_ori_test = data_ori_test2
    label = []
    for idx2 in range(len(data_ori_test)):
        one = data_ori_test[idx2]
        if tasks[idx] == 'stance_clean':
            label.append(CONV[one['labels']])
        else:
            label.append(int(one['labels']))

    data_si = datasets.load_from_disk('../finetune/data/' + tasks[idx] + '/hash_fulldata_simcse_top_1')
    data_si_test = data_si['test']
    data_si_test2 = []
    for one in data_si_test:
        data_si_test2.append(one)
    data_si_test = data_si_test2

    data_re = datasets.load_from_disk('../finetune/data/' + tasks[idx] + '/hash_fuldata_bt_hashseg_top_1')
    data_re_test = data_re['test']
    data_re_test2 = []
    for one in data_re_test:
        data_re_test2.append(one)
    data_re_test = data_re_test2

    ITER = 0
    for idx2 in range(len(label)):
        # if preds[tasks[idx]][ITER][idx2] != label[idx2]\
        #     and preds_si[tasks[idx]][ITER][idx2] != label[idx2]\
        #     and preds_re[tasks[idx]][ITER][idx2] == label[idx2]:

        if label[idx2] != 1:
            continue
        print(tasks[idx] +','+ str(label[idx2]))
        print(tokenizer.decode(data_ori_test[idx2]['input_ids']))
        # print(tokenizer.decode(data_re_test[pairs_hashseg[tasks[idx]][idx2]]['input_ids'][0]))
        print(tokenizer.decode(data_re_test[pairs_hashseg[tasks[idx]][idx2]]['input_ids'][1]))

        # print(tokenizer.decode(data_si_test[pairs_simcse[tasks[idx]][idx2]]['input_ids'][0]))
        print(tokenizer.decode(data_si_test[pairs_simcse[tasks[idx]][idx2]]['input_ids'][1]))