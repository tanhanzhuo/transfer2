import os.path
import numpy as np
from scipy.stats import ttest_ind

def read_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines1 = f.readlines()
    results1 = {}
    cur_task = ''
    for line in lines1:
        line_sp = line.split(':')
        if line_sp[0] == 'Task':
            cur_task = line_sp[1].split(',')[0].strip()
            results1[cur_task] = []
        elif line_sp[0] == 'aveRec':
            continue
        elif len(line)>5:
            num = float(line.split(',')[0])
            results1[cur_task].append(num)
    for task in results1.keys():
        scores = results1[task]
        new_scores = []
        me = np.median(scores)
        for score in scores:
            if score < me*0.9:
                continue
            else:
                new_scores.append(score)
        results1[task] = new_scores

    return  results1

# re_ori = read_txt('results_811_lrtune_full_lr1e5.txt')
# re_roberta = read_txt('results_811_lrtune_full_lr1e5_roberta.txt')
# re_time = read_txt('results_811_lrtune_full_lr1e5_extend_type1.txt')
re_ori = read_txt('results_811_lrtune_full_lr1e5.txt')
re_roberta = read_txt('results_811_lrtune_full_lr1e5_roberta.txt')
re_time = read_txt('results_811_lrtune_full_lr1e5_extend_type1.txt')

END = -2
count = {}
for task in re_roberta.keys():
    if task[:END] not in count.keys():
        count[task[:END]] = [0,0]
    st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
    if np.mean(re_ori[task]) > np.mean(re_roberta[task]):
        count[task[:END]][0] += 1
        if val_ro < 0.05:
            count[task[:END]][1] += 1
for task in count.keys():
    print('task:{}, larger:{},significant:{}'.format(task,count[task][0],count[task][1]))


for task in 'eval-stance_clean_811_8,eval-emotion_clean_811_5,eval-irony_clean_811_3,eval-offensive_clean_811_0,eval-hate_clean_811_3,sem21-task7-humor_clean_811_2,sem22-task6-sarcasm_clean_811_2,stance_clean_811_0'.split(','):
    st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
    st2, val_ro2 = ttest_ind(re_ori[task], re_time[task])
    print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
          .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]),
                  np.std(re_roberta[task]),
                  np.mean(re_time[task]), np.std(re_time[task])))
    print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task, val_ro, val_ro2))

# for task in re_ori.keys():
#     st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
#     # st2, val_ro2 = ttest_ind(re_ori[task], re_time[task])
#     # print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
#     #       .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]),
#     #               np.std(re_roberta[task]),
#     #               np.mean(re_time[task]), np.std(re_time[task])))
#     # print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task, val_ro, val_ro2))
#     if task[:END] == 'eval-offensive_clean_811' or task[:END] == 'stance_clean_811' or task[:END] == 'eval-irony_clean_811'\
#             or task[:END] == 'stance_clean_811_':
#         st2, val_ro2 = ttest_ind(re_ori[task], re_time[task])
#         if np.mean(re_ori[task]) < np.mean(re_time[task]) and val_ro2 < 0.05:
#             print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
#                   .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]), np.std(re_roberta[task]),
#                           np.mean(re_time[task]), np.std(re_time[task])))
#             print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task, val_ro, val_ro2))
#
#     else:
#         if np.mean(re_ori[task]) > np.mean(re_roberta[task]) and val_ro < 0.05:
#             st2, val_ro2 = ttest_ind(re_ori[task], re_time[task])
#
#         # st, val_ti = ttest_ind(re_ori[task], re_time[task])
#         # print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}'\
#         #       .format(task, np.mean(re_ori[task]),np.std(re_ori[task]), np.mean(re_roberta[task]),np.std(re_roberta[task]),np.mean(re_time[task]),np.std(re_roberta[task])))
#         # print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task,val_ro,val_ti))
#
#         #
#         # print('task:{}, bertweet:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
#         #       .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_time[task]),
#         #               np.std(re_time[task])))
#         # print('task:{}, vs roberta:{:.5f}'.format(task, val_ro2))
#             if np.mean(re_ori[task]) < np.mean(re_time[task]) and val_ro2 < 0.05:
#                 print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
#                       .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]), np.std(re_roberta[task]),
#                               np.mean(re_time[task]), np.std(re_time[task])))
#                 print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task, val_ro, val_ro2))
