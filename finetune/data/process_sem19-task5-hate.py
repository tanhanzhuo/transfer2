import pandas as pd
import json
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)

for SP in ['train','dev','test']:
    data_file = pd.read_csv("../data_raw/sem19-task5-hate/hateval2019_en_" + SP + ".csv", delimiter=',')
    data = []
    for idx in range(data_file.__len__()):
        data.append(str(data_file['HS'][idx]) + '\t' + data_file['text'][idx] + '\n')
    random.shuffle(data)
    write_txt('./sem19-task5-hate/'+ SP, data)
