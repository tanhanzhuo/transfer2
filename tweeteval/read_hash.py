import json

import datasets
import numpy as np

hash_samples = []
hash_tags = []
for idx in range(4):
    tmp = np.load('../contrastive_full/feature_modelT100N100M_fileT100N100S_num10_cluster'+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.extend(tmp['center_samples'])
    hash_tags.extend(tmp['center_hash'])
    tmp.close()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base',normalization=True)


tasks = 'eval-stance_clean_811_8,eval-emotion_clean_811_5,eval-irony_clean_811_3,eval-offensive_clean_811_0,eval-hate_clean_811_3,sem21-task7-humor_clean_811_2,sem22-task6-sarcasm_clean_811_2,stance_clean_811_0'.split(',')

pairs = {}
for idx in range(len(tasks)):
    data_ori = datasets.load_from_disk('../finetune/data/'+tasks[idx]+'/token')
    data_ori_test = data_ori['test']

    with open('../finetune/data/'+tasks[idx]+'/test_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst.json', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data_json = []
        for line in lines:
            data_json.append(json.loads(line))
    data_json_token = []
    for data_one in data_json:
        data_one_text = data_one['text'].split(' \n ')[-2]
        data_one_token = tokenizer(data_one_text, truncation=True, max_length=128)['input_ids']
        data_json_token.append(data_one_token)
    pair = {}
    for idx1 in range(len(data_ori_test)):
        one = data_ori_test[idx1]
        one_in = one['input_ids']
        for idx2 in range(len(data_json_token)):
            two = data_json_token[idx2]
            if one_in == two:
                retrieve_text = data_json[idx2]['text'].split(' \n ')[0].strip()
                idx3 = hash_samples.index(retrieve_text)
                retrieve_hash = hash_tags[int(idx3/10)]
                pair[idx1] = retrieve_hash
        if idx1 not in pair.keys():
            print('error!!!idx:{}'.format(idx1))
    pairs[tasks[idx]] = pair

import pickle
with open('pairs_hash.pickle', 'wb') as handle:
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('pairs.pickle', 'rb') as handle:
#     pairs = pickle.load(handle)
