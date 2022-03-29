import json
import os
path = '/work/test/finetune_newdata/'
new = 'data/complaint/'
if not os.path.isdir(path+new):
    os.mkdir(path+new)
old = 'data_raw/complaint/'

data_all = []
with open(path+old + 'complaint_severity_data.csv', 'r') as f:
    for line in f:
        line_sp = line.split(',')
        if len(line_sp) != 5 or int(line_sp[3]) == 0:
            continue
        one = line_sp[3] + '\t' + line_sp[1].replace('<user>', '@USER').replace('<url>', 'HTTPURL') + '\n'
        data_all.append(one)

import random
LEN = len(data_all)
SP = int(LEN*0.1)
for TIME in range(10):
    idx_all = list(range(LEN))
    
    random.shuffle(idx_all)
    with open(path+new+ 'train_' + str(TIME), 'w') as f_w:
        for idx in idx_all[:SP*8]:
            one = data_all[idx]
            f_w.write(one)

    with open(path+new+ 'dev_' + str(TIME), 'w') as f_w:
        for idx in idx_all[SP*8:SP*9]:
            one = data_all[idx]
            f_w.write(one)

    with open(path+new+ 'test_' + str(TIME), 'w') as f_w:
        for idx in idx_all[9*SP:]:
            one = data_all[idx]
            f_w.write(one)