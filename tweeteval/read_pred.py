import json
import matplotlib.pyplot as plt
import datasets
import numpy as np
import argparse
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

with open('pairs_hash_fuldata_bt_hashseg.pickle', 'rb') as handle:
    pairs = pickle.load(handle)

# with open('pairs_hash_fulldata_simcse.pickle', 'rb') as handle:
#     pairs = pickle.load(handle)

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

# tasks = 'eval-stance_clean_811_8,eval-emotion_clean_811_5,eval-irony_clean_811_3,eval-offensive_clean_811_0,eval-hate_clean_811_3,sem21-task7-humor_clean_811_2,sem22-task6-sarcasm_clean_811_2,stance_clean_811_0'.split(',')
tasks = 'eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean,stance_clean'.split(',')
# tasks = 'eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean'.split(',')
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
                tmp_reindex.append(int(tmp[pairs[cur_task][idx]]))
            preds_re[cur_task].append(tmp_reindex)



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
    data_ori = datasets.load_from_disk('../finetune/data/'+tasks[idx]+'/token')
    data_ori_test = data_ori['test']
    label = []
    length = []
    number = []
    for idx2 in range(len(data_ori_test)):
        one = data_ori_test[idx2]
        if tasks[idx] == 'stance_clean':
            label.append(CONV[one['labels']])
        else:
            label.append(int(one['labels']))
        length.append(len(one['input_ids'])-2)
        # number.append(hash_dic[pairs_hash[tasks[idx]][idx2]])

    ##### length
    pred_len_sp_label = []
    for idx3 in range(len(len_thre)+1):
        pred_len_sp_label.append([])

    pred_len_sp_pred = []
    for idx3 in range(len(preds[tasks[idx]])):
        pred_len_sp_pred.append([])
        for idx4 in range(len(len_thre)):
            pred_len_sp_pred[idx3].append([])

    pred_len_sp_pred_re = []
    for idx3 in range(len(preds_re[tasks[idx]])):
        pred_len_sp_pred_re.append([])
        for idx4 in range(len(len_thre)):
            pred_len_sp_pred_re[idx3].append([])

    for idx3 in range(len(length)):
        val = length[idx3]
        for idx_thre in range(len(len_thre)):
            thre = len_thre[idx_thre]
            if val < thre:
                pred_len_sp_label[idx_thre].append(label[idx3])
                for idx4 in range(len(preds[tasks[idx]])):
                    pred_len_sp_pred[idx4][idx_thre].append( preds[tasks[idx]][idx4][idx3] )
                for idx4 in range(len(preds_re[tasks[idx]])):
                    pred_len_sp_pred_re[idx4][idx_thre].append( preds_re[tasks[idx]][idx4][idx3] )
                break
    re_plot = [[],[],[]]
    for idx_thre in range(len(len_thre)):
        lab = []
        pre = []
        for idx4 in range(len(preds[tasks[idx]])):
            if len(pred_len_sp_label[idx_thre]) < 10:
                continue
            lab += pred_len_sp_label[idx_thre]
            pre += pred_len_sp_pred[idx4][idx_thre]
        if len(lab) < 1:
            continue
        result = cal_metric(lab,pre,tasks[idx])
        print('task:{}, length:{}, ori:{:.5f}'.format(tasks[idx],len_thre[idx_thre],result))
        len_result[idx_thre][0].append(result)

        re_plot[0].append(len_thre[idx_thre])
        re_plot[1].append(result)

        lab = []
        pre = []
        for idx4 in range(len(preds_re[tasks[idx]])):
            if len(pred_len_sp_label[idx_thre]) < 10:
                continue
            lab += pred_len_sp_label[idx_thre]
            pre += pred_len_sp_pred_re[idx4][idx_thre]
        if len(lab) < 1:
            continue
        result = result = cal_metric(lab,pre,tasks[idx])
        print('task:{}, length:{}, ret:{:.5f}'.format(tasks[idx],len_thre[idx_thre],result))
        len_result[idx_thre][1].append(result)

        re_plot[2].append(result)
    # plt.figure()
    # plt.plot(re_plot[0],re_plot[1])
    # plt.plot(re_plot[0], re_plot[2])
    # plt.suptitle(tasks[idx])
    # plt.show()




    # ###### hash number
    # pred_num_sp_label = []
    # for idx3 in range(len(num_thre)+1):
    #     pred_num_sp_label.append([])
    #
    # pred_num_sp_pred = []
    # for idx3 in range(len(preds[tasks[idx]])):
    #     pred_num_sp_pred.append([])
    #     for idx4 in range(len(num_thre)):
    #         pred_num_sp_pred[idx3].append([])
    #
    # pred_num_sp_pred_re = []
    # for idx3 in range(len(preds_re[tasks[idx]])):
    #     pred_num_sp_pred_re.append([])
    #     for idx4 in range(len(num_thre)):
    #         pred_num_sp_pred_re[idx3].append([])
    #
    # for idx3 in range(len(length)):
    #     val = number[idx3]
    #     for idx_thre in range(len(num_thre)):
    #         thre = num_thre[idx_thre]
    #         if val < thre:
    #             pred_num_sp_label[idx_thre].append(label[idx3])
    #             for idx4 in range(len(preds[tasks[idx]])):
    #                 pred_num_sp_pred[idx4][idx_thre].append( preds[tasks[idx]][idx4][idx3] )
    #             for idx4 in range(len(preds_re[tasks[idx]])):
    #                 pred_num_sp_pred_re[idx4][idx_thre].append( preds_re[tasks[idx]][idx4][idx3] )
    #             break
    # re_plot = [[],[],[]]
    # for idx_thre in range(len(num_thre)):
    #     lab = []
    #     pre = []
    #     for idx4 in range(len(preds[tasks[idx]])):
    #         if len(pred_num_sp_label[idx_thre]) < 10:
    #             continue
    #         lab += pred_num_sp_label[idx_thre]
    #         pre += pred_num_sp_pred[idx4][idx_thre]
    #     if len(lab) < 1:
    #         continue
    #     result = cal_metric(lab,pre,tasks[idx])
    #     print('task:{}, length:{}, ori:{:.5f}'.format(tasks[idx],num_thre[idx_thre],result))
    #     num_result[idx_thre][0].append(result)
    #
    #     re_plot[0].append(num_thre[idx_thre])
    #     re_plot[1].append(result)
    #
    #     lab = []
    #     pre = []
    #     for idx4 in range(len(preds_re[tasks[idx]])):
    #         if len(pred_num_sp_label[idx_thre]) < 10:
    #             continue
    #         lab += pred_num_sp_label[idx_thre]
    #         pre += pred_num_sp_pred_re[idx4][idx_thre]
    #     if len(lab) < 1:
    #         continue
    #     result = result = cal_metric(lab,pre,tasks[idx])
    #     print('task:{}, length:{}, ret:{:.5f}'.format(tasks[idx],num_thre[idx_thre],result))
    #     num_result[idx_thre][1].append(result)
    #
    #     re_plot[2].append(result)
    # plt.figure()
    # plt.plot(re_plot[0],re_plot[1])
    # plt.plot(re_plot[0], re_plot[2])
    # plt.suptitle(tasks[idx])
    # plt.show()

#
# for idx_thre in range(len(len_thre)):
#     cur_re = len_result[idx_thre]
#     if len(cur_re[0]) < 1:
#         continue
#     print('length:{}, ori:{:.5f}, ret:{:.5f}'.format(len_thre[idx_thre], np.average(cur_re[0]),np.average(cur_re[1])))
#
for idx_thre in range(len(len_thre)):
    cur_re = len_result[idx_thre]
    if len(cur_re[0]) < 1:
        continue
    print('{:.5f},{:.5f}'.format(np.average(cur_re[0]),np.average(cur_re[1])))