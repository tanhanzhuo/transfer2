import datasets
import numpy as np
import argparse
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

with open('pairs.pickle', 'rb') as handle:
    pairs = pickle.load(handle)
parser = argparse.ArgumentParser()
parser.add_argument("--ori", default='ori_pred.txt', type=str, required=False)
parser.add_argument("--ret", default='ret_pred.txt', type=str, required=False)
args = parser.parse_args()

def cal_metric(label_all,pred_all,task='eval-emoji'):
    results = classification_report(label_all, pred_all, output_dict=True)

    if 'emoji' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Emotion (Macro f1)
    elif 'emotion' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Hate (Macro f1)
    elif 'hate' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Irony (Irony class f1)
    elif 'irony' in task:
        tweeteval_result = results['1']['f1-score']

        # Offensive (Macro f1)
    elif 'offensive' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Sentiment (Macro Recall)
    elif 'sentiment' in task:
        tweeteval_result = results['macro avg']['recall']

        # Stance (Macro F1 of 'favor' and 'against' classes)
    elif 'stance' in task:
        f1_against = results['1']['f1-score']
        f1_favor = results['2']['f1-score']
        tweeteval_result = (f1_against + f1_favor) / 2
    elif 'sarcasm' in task:
        tweeteval_result = results['1']['f1-score']
    elif 'humor' in task:
        tweeteval_result = results['1']['f1-score']
    return tweeteval_result

tasks = 'eval-stance_clean_811_8,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance'.split(',')
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
                tmp_reindex.append(tmp[pairs[cur_task][idx]])
            preds_re[cur_task].append(tmp_reindex)

for idx in range(len(tasks)):
    data_ori = datasets.load_from_disk('../finetune/data/'+tasks[idx]+'/token')
    data_ori_test = data_ori['test']
    label = []
    length = []
    for one in data_ori_test:
        label.append(one['labels'])
        length.append(len(one['input_ids']))

