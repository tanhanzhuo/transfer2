import json
import os
path = '/work/test/finetune_newdata/'
new = 'data/amazon-review-helpful/'
if not os.path.isdir(path+new):
    os.mkdir(path+new)
old = 'data_raw/amazon-review-helpful/'

# with open(path+new+ 'train.txt', 'w') as f_w:
#     with open(path+old + 'train.json', 'r') as f:
#         for line in f:
#             one = json.loads(line)
#             f_w.write(str(one['helpful']) + '\t' + one['sentence'] + '\n')

data_all = []
with open(path+old + 'train.json', 'r') as f:
    for line in f:
        one = json.loads(line)
        data_all.append(one)
            
import random
LEN = len(data_all)
idx_all = list(range(LEN))
TRAIN = int(LEN*0.9)
random.shuffle(idx_all)
with open(path+new+ 'train', 'w') as f_w:
    for idx in idx_all[:TRAIN]:
        one = data_all[idx]
        f_w.write(str(one['helpful']) + '\t' + one['sentence'] + '\n')

with open(path+new+ 'dev', 'w') as f_w:
    for idx in idx_all[TRAIN:]:
        one = data_all[idx]
        f_w.write(str(one['helpful']) + '\t' + one['sentence'] + '\n')
