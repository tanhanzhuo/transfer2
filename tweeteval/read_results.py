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

# re_ori = read_txt('results_lr1e5_clean2.txt')
# # re_btcluster = read_txt('results_lr1e5_clean_extend_type1_random4.txt')
# re_fullbt = read_txt('results_lr1e5_clean_extend_type1_fullbt_seg2.txt')
# re_fullsimcse = read_txt('results_lr1e5_clean_extend_type1_fullsimcse2.txt')
# tmp=1


# re_ori = read_txt('results_811_lrtune_full_lr1e5.txt')
# re_roberta = read_txt('results_811_lrtune_full_lr1e5_roberta.txt')
# re_time = read_txt('results_811_lrtune_full_lr1e5_extend_type1.txt')

# for task in 'eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance'.split(','):
#     print('{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}' \
#           .format(np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]),
#                   np.std(re_roberta[task]),np.mean(re_time[task]), np.std(re_time[task])))

# for task in 'eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean,stance_clean'.split(','):
#     print('{:.5f},{:.5f},{:.5f},{:.5f}' \
#           .format(np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_fullbt[task]),
#                   np.std(re_fullbt[task])))


# re_ori.pop('stance_clean')
# re_fullbt.pop('stance_clean')
# re_fullsimcse.pop('stance_clean')
# ori_values = np.mean(list(re_ori.values()),axis=0)
# bt_values = np.mean(list(re_fullbt.values()),axis=0)
# sim_values = np.mean(list(re_fullsimcse.values()),axis=0)
#
# st, val_ro = ttest_ind(ori_values, bt_values)


# for task in 'eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean,stance_clean'.split(','):
#     print('{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}' \
#           .format(np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_btcluster[task]),np.std(re_btcluster[task]), \
#                   np.mean(re_fullbt[task]), np.std(re_fullbt[task]), np.mean(re_fullsimcse[task]), np.std(re_fullsimcse[task])
#                   )
#           )

# for top in [2,3,5,7,9,11,13,15,17,19]:
#     re_fullbt = read_txt('results_lr1e5_clean_extend_type1_fullbt_seg_top'+str(top)+'1.txt')
#     re_fullsimcse = read_txt('results_lr1e5_clean_extend_type1_fullsimcse_top'+str(top)+'1.txt')
#
#     re_fullbt.pop('stance_clean')
#     re_fullsimcse.pop('stance_clean')
#     print('{:.5f},{:.5f}'.format(np.mean([np.mean(i) for i in list(re_fullbt.values())]),\
#                                  np.mean([np.mean(i) for i in list(re_fullsimcse.values())])\
#                                  ))

re_ori = read_txt('results_811_lrtune_full_lr1e5.txt')
# re_btcluster = read_txt('results_lr1e5_clean_extend_type1_random4.txt')
re_roberta = read_txt('results_811_lrtune_full_lr1e5_roberta.txt')
re_bt = read_txt('results_811_lrtune_full_lr1e5_extend_type1.txt')
re_sim = read_txt('results_811_lrtune_full_lr1e5_extend_type1_simcsefull_verify.txt')
# re_ori_flat = {}
# re_roberta_flat = {}
# END = -6
# count = {}
# for task in re_roberta.keys():
#     if task[:END] not in re_ori_flat.keys():
#         re_ori_flat[task[:END]] = []
#     re_ori_flat[task[:END]].extend(re_ori[task])
#     if task[:END] not in re_roberta_flat.keys():
#         re_roberta_flat[task[:END]] = []
#     re_roberta_flat[task[:END]].extend(re_roberta[task])
#
# for task in 'eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean'.split(','):
#     print('{:.5f},{:.5f}' \
#           .format(np.mean(re_ori_flat[task]), np.mean(re_roberta_flat[task])))

for task in 'eval-stance_clean_811_8,eval-emotion_clean_811_5,eval-irony_clean_811_3,eval-offensive_clean_811_0,eval-hate_clean_811_3,sem21-task7-humor_clean_811_2,sem22-task6-sarcasm_clean_811_2'.split(','):
    # st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
    # print('{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}' \
    #       .format(np.mean(re_ori[task]), np.std(re_ori[task]), \
    #             np.mean(re_roberta[task]), np.std(re_roberta[task]),
    #               val_ro
    #               ))

    st, val_ro = ttest_ind(re_ori[task], re_bt[task])
    st1, val_ro1 = ttest_ind(re_ori[task], re_sim[task])
    print('{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}' \
          .format(np.mean(re_bt[task]), np.std(re_bt[task]), val_ro,\
                  np.mean(re_sim[task]), np.std(re_sim[task]),val_ro1
                  ))



# for task in count.keys():
#     print('task:{}, larger:{},significant:{}'.format(task,count[task][0],count[task][1]))


# for task in 'eval-stance_clean_811_8,eval-emotion_clean_811_5,eval-irony_clean_811_3,eval-offensive_clean_811_0,eval-hate_clean_811_3,sem21-task7-humor_clean_811_2,sem22-task6-sarcasm_clean_811_2,stance_clean_811_0'.split(','):
#     st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
#     st2, val_ro2 = ttest_ind(re_ori[task], re_time[task])
#     print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
#           .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]),
#                   np.std(re_roberta[task]),
#                   np.mean(re_time[task]), np.std(re_time[task])))
#     print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task, val_ro, val_ro2))



# for task in re_ori.keys():
#     st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
#     st2, val_ro2 = ttest_ind(re_ori[task], re_time[task])
#     print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}' \
#           .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]),
#                   np.std(re_roberta[task]),
#                   np.mean(re_time[task]), np.std(re_time[task])))
#     print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task, val_ro, val_ro2))

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
