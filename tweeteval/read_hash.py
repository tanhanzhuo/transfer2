import datasets
import numpy as np

hash_samples = []
hash_tags = []
for idx in range(4):
    tmp = np.load('feature_modelT100N100M_fileT100N100S_num10_cluster'+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.append(tmp['center_samples'])
    hash_tags.append(tmp['center_hash'])
    tmp.close()



tasks = 'eval-stance_clean_811_8,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance'.split(',')

pairs = {}
for idx in range(len(tasks)):
    data_ori = datasets.load_from_disk('../finetune/data/'+tasks[idx]+'/token')
    data_ori_test = data_ori['test']

    data_ex = datasets.load_from_disk('../finetune/data/' + tasks[idx] + '/hash_modelT100N100M_fileT100N100S_num10_cluster_top_1')
    data_ex_test = data_ex['test']

    pair = {}
    for idx1 in range(len(data_ori_test)):
        one = data_ori_test[idx1]
        one_in = one['input_ids']
        for idx2 in range(len(data_ex_test)):
            two = data_ex_test[idx2]
            two_in = two['input_ids'][0]
            if one_in == two_in:
                pair[idx1] = idx2
        if idx1 not in pair.keys():
            print('error!!!idx:{}'.format(idx1))
    pairs[tasks[idx]] = pair

import pickle
with open('pairs.pickle', 'wb') as handle:
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pairs.pickle', 'rb') as handle:
#     pairs = pickle.load(handle)
