import pandas as pd
data_file = pd.read_csv("../data_raw/sem19-task6-offen/olid-training-v1.0.tsv", delimiter='\t')

data = []
for idx in range(data_file.__len__()):
    data.append(data_file['subtask_a'][idx] + '\t' + data_file['tweet'][idx] + '\n')

import json
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)
random.shuffle(data)
SP = int(len(data)/10)
write_txt('./sem19-task6-offen2/train', data[:9*SP])
write_txt('./sem19-task6-offen2/dev', data[9*SP:])

data_file = pd.read_csv("../data_raw/sem19-task6-offen/testset-levela.tsv", delimiter='\t')
data_lab = pd.read_csv("../data_raw/sem19-task6-offen/labels-levela.csv", delimiter=',', names=['id', 'lab'])
data = []
for idx in range(data_file.__len__()):
    data.append(data_lab['lab'][idx] + '\t' + data_file['tweet'][idx] + '\n')
write_txt('./sem19-task6-offen2/test', data)