import os.path
import random
import json

for epoch in range(10):
    for task in ['emoji','emotion','hate','irony','offensive','sentiment',\
                     'stance/abortion','stance/atheism','stance/climate','stance/feminist','stance/hillary']:
        data_tmp = []
        for sp in ['train','val']:
            if not os.path.isdir('eval-'+task+'_trainsplit'+str(epoch)):
                os.mkdir('eval-'+task+'_trainsplit'+str(epoch))
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
            val_num = len(lines_text)
        random.shuffle(data_tmp)
        total_num = len(data_tmp)

        with open('./eval-' + task + '_trainsplit'+str(epoch)+'/' + 'dev.json', 'w', encoding='utf-8') as f:
            for idx in range(0,val_num):
                one = data_tmp[idx]
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')
        with open('./eval-' + task + '_trainsplit'+str(epoch)+'/' + 'train.json', 'w', encoding='utf-8') as f:
            for idx in range(val_num, total_num):
                one = data_tmp[idx]
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')

        data_tmp = []
        for sp in ['test']:
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
        with open('./eval-' + task + '_trainsplit'+str(epoch)+'/' + 'test.json', 'w', encoding='utf-8') as f:
            for idx in range(len(data_tmp)):
                one = data_tmp[idx]
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')