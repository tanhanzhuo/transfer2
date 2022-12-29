import os.path
import random
import json

# for epoch in range(10):
#     # for task in ['emoji','emotion','hate','irony','offensive','sentiment',\
#     #                  'stance/abortion','stance/atheism','stance/climate','stance/feminist','stance/hillary']:
#     for task in ['emotion']:#['hate', 'irony', 'offensive']:
#         data_tmp = []
#         for sp in ['train','val','test']:
#             if not os.path.isdir('eval-'+task+'_evensplit'+str(epoch)):
#                 os.mkdir('eval-'+task+'_evensplit'+str(epoch))
#             with open('../data_raw/eval-'+task+'/'+sp+'_text.txt','r',encoding='utf-8') as f:
#                 lines_text = f.readlines()
#             with open('../data_raw/eval-' + task + '/' + sp + '_labels.txt', 'r', encoding='utf-8') as f:
#                 lines_labels = f.readlines()
#
#             for idx in range(len(lines_text)):
#                 line_text = lines_text[idx].strip()
#                 line_label = lines_labels[idx].strip()
#                 if len(line_text)<1:
#                     continue
#                 one = {'labels':line_label,'text':line_text}
#                 data_tmp.append(one)
#         random.shuffle(data_tmp)
#         total_num = len(data_tmp)
#         split_num = int(total_num/3)
#
#         with open('./eval-' + task + '_evensplit'+str(epoch)+'/' + 'train.json', 'w', encoding='utf-8') as f:
#             for idx in range(0,split_num):
#                 one = data_tmp[idx]
#                 tmp = json.dumps(one, ensure_ascii=False)
#                 f.write(tmp + '\n')
#         with open('./eval-' + task + '_evensplit'+str(epoch)+'/' + 'dev.json', 'w', encoding='utf-8') as f:
#             for idx in range(split_num, split_num*2):
#                 one = data_tmp[idx]
#                 tmp = json.dumps(one, ensure_ascii=False)
#                 f.write(tmp + '\n')
#         with open('./eval-' + task + '_evensplit'+str(epoch)+'/' + 'test.json', 'w', encoding='utf-8') as f:
#             for idx in range(split_num*2, total_num):
#                 one = data_tmp[idx]
#                 tmp = json.dumps(one, ensure_ascii=False)
#                 f.write(tmp + '\n')



####### semeval22 sarcasm
import pandas as pd
import json

for epoch in range(10):
    for task in ['sarcasm']:
        if not os.path.isdir('eval-' + task + '_evensplit' + str(epoch)):
            os.mkdir('eval-' + task + '_evensplit' + str(epoch))
        data = []
        data_file = pd.read_csv("../data_raw/sem22-task6-sarcasm/train.En.csv", delimiter=',')
        for idx in range(data_file.__len__()):
            if data_file['tweet'][idx] == data_file['tweet'][idx]:
                # data.append(str(data_file['sarcastic'][idx]) + '\t' + data_file['tweet'][idx].replace('\n','') + '\n')
                one = {'labels': str(data_file['sarcastic'][idx]), 'text': data_file['tweet'][idx].replace('\n','')}
                data.append(one)

        data_file = pd.read_csv("../data_raw/sem22-task6-sarcasm/task_A_En_test.csv", delimiter=',')
        for idx in range(data_file.__len__()):
            # data.append(str(data_file['sarcastic'][idx]) + '\t' + data_file['text'][idx].replace('\n','') + '\n')
            one = {'labels': str(data_file['sarcastic'][idx]), 'text': data_file['text'][idx].replace('\n', '')}
            data.append(one)

        random.shuffle(data)
        total_num = len(data)
        split_num = int(total_num/3)
        with open('./eval-' + task + '_evensplit' + str(epoch) + '/' + 'train.json', 'w', encoding='utf-8') as f:
            for idx in range(0, split_num):
                one = data[idx]
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')
        with open('./eval-' + task + '_evensplit' + str(epoch) + '/' + 'dev.json', 'w', encoding='utf-8') as f:
            for idx in range(split_num, split_num * 2):
                one = data[idx]
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')
        with open('./eval-' + task + '_evensplit' + str(epoch) + '/' + 'test.json', 'w', encoding='utf-8') as f:
            for idx in range(split_num * 2, total_num):
                one = data[idx]
                tmp = json.dumps(one, ensure_ascii=False)
                f.write(tmp + '\n')