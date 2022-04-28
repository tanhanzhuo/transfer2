import random
import pandas as pd
import os
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)

data_all = []
name_all = ['face_masks', 'fauci', 'school_closures', 'stay_at_home_orders']
for idx in range(len(name_all)):
    name = name_all[idx]
    data_all.append([])
    for sp in ['train', 'val', 'test']:
        data = pd.read_csv('../data_raw/stance/' + name + '_' + sp + '.csv', sep=',')
        for idx2 in range(data.__len__()):
            data_all[idx].append(data.Stance[idx2] + '\t' + data.Tweet[idx2].replace('\n', ' ').replace('\t', '') + '\n')

for idx_test in range(len(name_all)):
    data_test = data_all[idx_test]
    data_train = []
    for idx_train in range(len(name_all)):
        if idx_train != idx_test:
            data_train.extend(data_all[idx_train])
    random.shuffle(data_train)
    random.shuffle(data_test)
    name = name_all[idx_test] + '_sp'
    dev_len = int( len(data_train)*0.2 )
    os.makedirs('./stance/'+name, exist_ok=True)
    write_txt('./stance/'+name+'/train', data_train[dev_len:])
    write_txt('./stance/'+name+'/dev', data_train[:dev_len])
    write_txt('./stance/'+name+'/test', data_test)