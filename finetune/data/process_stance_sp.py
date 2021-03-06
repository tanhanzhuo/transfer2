# import os
# for name in ['face_masks', 'fauci', 'school_closures', 'stay_at_home_orders']:
#     if not os.path.isdir('/work/test/finetune_newdata/data/stance/' + name):
#         os.mkdir('/work/test/finetune_newdata/data/stance/' + name)
#     for sp in ['train', 'val', 'test']:
#         with open('/work/test/finetune_newdata/data_raw/stance/' + name + '_' + sp + '.csv', 'r') as f:
#             lines = f.readlines()
#         if sp == 'val':
#             sp_save = 'dev'
#         else:
#             sp_save = sp
#         with open('/work/test/finetune_newdata/data/stance/' + name + '/' + sp_save , 'w') as f:
#             for line in lines:
#                 line_sp = line.split('\t')
#                 f.write(str(line_sp[1]) + '\t' + line_sp[2] )

import pandas as pd
train_all = []
dev_all = []
test_all = []
for name in ['face_masks', 'fauci', 'school_closures', 'stay_at_home_orders']:
    for sp in ['train', 'val', 'test']:
        data = pd.read_csv('../data_raw/stance/' + name + '_' + sp + '.csv', sep=',')
        for idx in range(data.__len__()):
            if sp == 'val':
                sp_save = 'dev'
            else:
                sp_save = sp
            with open('./stance/' + name + '_new/' + sp_save, 'a', encoding='utf-8') as f:
                f.write(data.Stance[idx] + '\t' + data.Tweet[idx].replace('\n', ' ').replace('\t','') + '\n')