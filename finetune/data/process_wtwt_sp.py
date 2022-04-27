import json
import os.path
import random
def write_txt(fileName, data):
    with open(fileName, 'w', encoding='utf-8') as f:
        for one in data:
            f.write(one['stance'] + '\t' + one['text'].replace('\t', '').replace('\n', ' ') + '\n')
with open('../data_raw/wtwt/wtwt_with_text.json', 'r') as f:
    data = json.load(f)

random.shuffle(data)
SP = ['CVS_AET', 'CI_ESRX', 'ANTM_CI', 'AET_HUM', 'FOXA_DIS']
for SP_one in SP:
    train = []
    test = []
    for data_one in data:
        if data_one['merger'] == SP_one:
            test.append(data_one)
        else:
            train.append(data_one)
    dev_len = int( len(train)*0.2 )
    os.makedirs('./wtwt/'+SP_one, exist_ok=True)
    write_txt('./wtwt/'+SP_one+'/train', data[dev_len:])
    write_txt('./wtwt/'+SP_one+'/dev', train[:dev_len])
    write_txt('./wtwt/'+SP_one+'/test', test)
