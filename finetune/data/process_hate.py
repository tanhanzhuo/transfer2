import numpy as np
import json

with open('/work/test/finetune_newdata/data_raw/HateXplain/Data/dataset.json', 'r') as f:
    lines = json.load(f)

with open('/work/test/finetune_newdata/data_raw/HateXplain/Data/post_id_divisions.json', 'r') as f:
    divisions = json.load(f)

f_train = open('/work/test/finetune_newdata/data/hate_all/train', 'a')
f_dev = open('/work/test/finetune_newdata/data/hate_all/dev', 'a')
f_test = open('/work/test/finetune_newdata/data/hate_all/test', 'a')
bad_case = 0
for div in divisions.keys():
    for id in divisions[div]:
        if 1:
            annotators = lines[id]['annotators']
            txt = ' '.join( lines[id]['post_tokens'] ).replace('<user>','@USER')
            label_list = [0,0,0]
            for one in annotators:
                label = one['label']
                if label == "normal":
                    label_list[0]+=1
                elif label == "hatespeech":
                    label_list[1]+=1
                elif label == "offensive":
                    label_list[2]+=1
                else:
                    print(id+'*********bad*******')
            if sum(label_list)!=3:
                print(id+'*********lack*******')
            if label_list == [1,1,1]:
                bad_case+=1
                continue
            label_final = np.argmax(label_list)
            if div == 'train':
                f_train.write(str(label) + '\t' + txt + '\n')
            elif div == 'val':
                f_dev.write(str(label) + '\t' + txt + '\n')
            elif div == 'test':
                f_test.write(str(label) + '\t' + txt + '\n')
print(bad_case)