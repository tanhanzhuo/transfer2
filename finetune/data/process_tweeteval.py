import os.path
import random
import json

for sp in ['train','val','test']:
    data_tmp = []
    for task in ['emoji','emotion','hate','irony','offensive','sentiment',\
                 'stance/abortion','stance/atheism','stance/climate','stance/feminist','stance/hillary']:
        if not os.path.isdir('eval-'+task):
            os.mkdir('eval-'+task)
        with open('../data_raw/eval-'+task+'/'+sp+'_text.txt','r',encoding='utf-8') as f:
            lines_text = f.readlines()
        with open('../data_raw/eval-' + task + '/' + sp + '_labels.txt', 'r', encoding='utf-8') as f:
            lines_labels = f.readlines()
        if sp == 'val':
            sp_w = 'dev'
        else:
            sp_w = sp
        with open('./eval-'+task+'/'+sp_w+'.json','w',encoding='utf-8') as f:
            for idx in range(len(lines_text)):
                line_text = lines_text[idx].strip()
                line_label = lines_labels[idx].strip()
                one = {'labels':line_label,'text':line_text}
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')