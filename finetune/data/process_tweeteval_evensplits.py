import os.path
import random
import json

for task in ['emoji','emotion','hate','irony','offensive','sentiment',\
                 'stance/abortion','stance/atheism','stance/climate','stance/feminist','stance/hillary']:
    data_tmp = []
    for sp in ['train','val','test']:
        if not os.path.isdir('eval-'+task+'_evensplit'):
            os.mkdir('eval-'+task+'_evensplit')
        with open('../data_raw/eval-'+task+'/'+sp+'_text.txt','r',encoding='utf-8') as f:
            lines_text = f.readlines()
        with open('../data_raw/eval-' + task + '/' + sp + '_labels.txt', 'r', encoding='utf-8') as f:
            lines_labels = f.readlines()

        for idx in range(len(lines_text)):
            line_text = lines_text[idx].strip()
            line_label = lines_labels[idx].strip()
            if len(line_text)<1:
                continue
            one = {'labels':line_label,'text':line_text}
            data_tmp.append(one)
    random.shuffle(data_tmp)
    total_num = len(data_tmp)
    split_num = int(total_num/3)

    with open('./eval-' + task + '_evensplit/' + 'train.json', 'w', encoding='utf-8') as f:
        for idx in range(0,split_num):
            one = data_tmp[idx]
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')
    with open('./eval-' + task + '_evensplit/' + 'dev.json', 'w', encoding='utf-8') as f:
        for idx in range(split_num, split_num*2):
            one = data_tmp[idx]
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')
    with open('./eval-' + task + '_evensplit/' + 'test.json', 'w', encoding='utf-8') as f:
        for idx in range(split_num*2, total_num):
            one = data_tmp[idx]
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp + '\n')