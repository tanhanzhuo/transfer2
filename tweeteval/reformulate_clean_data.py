import json
import random
import os

for epoch in range(10):
    for task in ['eval-irony_clean','eval-offensive_clean','eval-emotion_clean','eval-stance_clean']:
        if not os.path.isdir('../finetune/data/' + task + '_evensplit'+str(epoch)):
            os.mkdir('../finetune/data/' + task + '_evensplit'+str(epoch))
        data_tmp = []
        with open('../finetune/data/' + task + '/all.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                one = json.loads(line)
                data_tmp.append(one)

            random.shuffle(data_tmp)
            total_num = len(data_tmp)
            split_num = int(total_num/3)


            with open('../finetune/data/' + task + '_evensplit'+str(epoch)+'/' + 'train.json', 'w', encoding='utf-8') as f:
                for idx in range(0,split_num):
                    one = data_tmp[idx]
                    tmp = json.dumps(one, ensure_ascii=False)
                    f.write(tmp + '\n')
            with open('../finetune/data/' + task + '_evensplit'+str(epoch)+'/' + 'dev.json', 'w', encoding='utf-8') as f:
                for idx in range(split_num, split_num*2):
                    one = data_tmp[idx]
                    tmp = json.dumps(one, ensure_ascii=False)
                    f.write(tmp + '\n')
            with open('../finetune/data/' + task + '_evensplit'+str(epoch)+'/' + 'test.json', 'w', encoding='utf-8') as f:
                for idx in range(split_num*2, total_num):
                    one = data_tmp[idx]
                    tmp = json.dumps(one, ensure_ascii=False)
                    f.write(tmp + '\n')