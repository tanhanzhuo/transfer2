import json

import datasets
import numpy as np

hash_samples = []
hash_tags = []
for idx in range(4):
    tmp = np.load('feature_modelT100N100M_fileT100N100S_num10_cluster'+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.append(tmp['center_samples'])
    hash_tags.append(tmp['center_hash'])
    tmp.close()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')


tasks = 'eval-stance_clean_811_8,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance'.split(',')

pairs = {}
for idx in range(len(tasks)):
    data_ori = datasets.load_from_disk('../finetune/data/'+tasks[idx]+'/token')
    data_ori_test = data_ori['test']

    with open('../finetune/data/'+tasks[idx]+'/test_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data_json = []
        for line in lines:
            data_json.append(json.loads(line))

    pair = {}
    for idx1 in range(len(data_ori_test)):
        one = data_ori_test[idx1]
        one_in = one['input_ids']
        for idx2 in range(len(data_json)):
            two = data_json[idx2]
            two_in = two['text'].split(' \n ')[-2]
            two_token = tokenizer(two_in, truncation=True,max_length=128)['input_ids']
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
