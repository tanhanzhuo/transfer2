import pandas as pd
import json
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)

data_file = pd.read_csv("../data_raw/sem22-task6-sarcasm/train.En.csv", delimiter=',')
data = []
for idx in range(data_file.__len__()):
    if data_file['tweet'][idx] == data_file['tweet'][idx]:
        data.append(str(data_file['sarcastic'][idx]) + '\t' + data_file['tweet'][idx].replace('\n','') + '\n')
random.shuffle(data)
SP = int(len(data)/10)
write_txt('./sem22-task6-sarcasm/train', data[:9*SP])
write_txt('./sem22-task6-sarcasm/dev', data[9*SP:])

data_file = pd.read_csv("../data_raw/sem22-task6-sarcasm/task_A_En_test.csv", delimiter=',')
data = []
for idx in range(data_file.__len__()):
    data.append(str(data_file['sarcastic'][idx]) + '\t' + data_file['text'][idx].replace('\n','') + '\n')
write_txt('./sem22-task6-sarcasm/test', data)