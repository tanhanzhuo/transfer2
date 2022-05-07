import pandas as pd
data_file = pd.read_csv("../data_raw/imp-hate/implicit_hate_v1_stg1_posts.tsv", delimiter='\t')

data = []
for idx in range(data_file.__len__()):
    if data_file['class'][idx] != 'explicit_hate':
        data.append(data_file['class'][idx] + '\t' + data_file['post'][idx] + '\n')

import json
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one)
random.shuffle(data)
SP = int(len(data)/10)
write_txt('./imp-hate2/train', data[:8*SP])
write_txt('./imp-hate2/dev', data[8*SP:9*SP])
write_txt('./imp-hate2/test', data[9*SP:])