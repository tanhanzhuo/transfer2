import random
import pandas as pd
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)

for sp in ['train','dev','gold-test-27446']:
    data_file = pd.read_csv("../data_raw/sem21-task7-humor/" + sp + ".csv", delimiter=',')
    data = []
    for idx in range(data_file.__len__()):
        data.append(str(data_file['is_humor'][idx]) + '\t' + data_file['text'][idx] + '\n')
    random.shuffle(data)
    if 'test' in sp:
        write_txt('./sem21-task7-humor/' + 'test', data)
    else:
        write_txt('./sem21-task7-humor/' + sp, data)