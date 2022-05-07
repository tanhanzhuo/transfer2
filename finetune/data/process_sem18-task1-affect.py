import pandas as pd
import json
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)

NAME = {'train':'train','dev':'dev','test-gold':'test'}
for SP in ['train','dev','test-gold']:
    data_file = pd.read_csv("../data_raw/sem18-task1-affect/V-oc/2018-Valence-oc-En-" + SP + ".txt", delimiter='\t')
    data = []
    for idx in range(data_file.__len__()):
        data.append(data_file['Intensity Class'][idx] + '\t' + data_file['Tweet'][idx] + '\n')
    random.shuffle(data)
    write_txt('./sem18-task1-affect/valence/'+ NAME[SP], data)

NAME = {'training':'train','development':'dev','test-gold':'test-gold'}
for EMO in ['anger', 'fear', 'joy', 'sadness']:
    for SP in ['training','development','test-gold']:
        if SP == 'training':
            data_file = pd.read_csv(
                "../data_raw/sem18-task1-affect/EI-oc/" + SP + '/EI-oc-En-' + EMO + '-' + NAME[SP] + ".txt",
                delimiter='\t')
        else:
            data_file = pd.read_csv(
                "../data_raw/sem18-task1-affect/EI-oc/" + SP + '/2018-EI-oc-En-' + EMO + '-' + NAME[SP] + ".txt",
                delimiter='\t')
        data = []
        for idx in range(data_file.__len__()):
            data.append(data_file['Intensity Class'][idx] + '\t' + data_file['Tweet'][idx] + '\n')
        random.shuffle(data)
        write_txt('./sem18-task1-affect/' + EMO + '/'+ NAME[SP].split('-')[0], data)