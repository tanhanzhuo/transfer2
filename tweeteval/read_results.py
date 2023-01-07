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

re_ori = read_txt('results_111_lrtune_full_lr1e5.txt')
re_roberta = read_txt('results_111_lrtune_full_lr1e5_extend_type1.txt')
# re_time = read_txt('results_ori_full_early_weight_time.txt')


for task in re_roberta.keys():
    st, val_ro = ttest_ind(re_ori[task], re_roberta[task])
    # st, val_ti = ttest_ind(re_ori[task], re_time[task])
    # print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}, time:{:.5f},std:{:.5f}'\
    #       .format(task, np.mean(re_ori[task]),np.std(re_ori[task]), np.mean(re_roberta[task]),np.std(re_roberta[task]),np.mean(re_time[task]),np.std(re_roberta[task])))
    # print('task:{}, vs roberta:{:.5f}, vs time:{:.5f}'.format(task,val_ro,val_ti))

    print('task:{}, bertweet:{:.5f},std:{:.5f}, roberta:{:.5f},std:{:.5f}' \
          .format(task, np.mean(re_ori[task]), np.std(re_ori[task]), np.mean(re_roberta[task]),
                  np.std(re_roberta[task])))
    print('task:{}, vs roberta:{:.5f}'.format(task, val_ro))