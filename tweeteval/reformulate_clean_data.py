import json
import random
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance',type=str)
parser.add_argument('--split',default='111',type=str)
args = parser.parse_args()

for epoch in range(10):
    for task in args.dataset.split(','):
        if not os.path.isdir('../finetune/data/' + task + '_clean_'+args.split+'_'+str(epoch)):
            os.mkdir('../finetune/data/' + task + '_clean_'+args.split+'_'+str(epoch))
        data_tmp = []
        with open('../finetune/data/' + task + '_clean/all.json', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                one = json.loads(line)
                data_tmp.append(one)

            random.shuffle(data_tmp)
            total_num = len(data_tmp)
            train_num = int(args.split[0])
            dev_num = int(args.split[1])
            test_num = int(args.split[2])
            split_num = int(total_num/(train_num+dev_num+test_num))

            with open('../finetune/data/' + task + '_clean_'+args.split+'_'+str(epoch)+'/' + 'train.json', 'w', encoding='utf-8') as f:
                for idx in range(0,split_num*train_num):
                    one = data_tmp[idx]
                    tmp = json.dumps(one, ensure_ascii=False)
                    f.write(tmp + '\n')
            with open('../finetune/data/' + task + '_clean_'+args.split+'_'+str(epoch)+'/' + 'dev.json', 'w', encoding='utf-8') as f:
                for idx in range(split_num*train_num, split_num*(train_num+dev_num)):
                    one = data_tmp[idx]
                    tmp = json.dumps(one, ensure_ascii=False)
                    f.write(tmp + '\n')
            with open('../finetune/data/'  + task + '_clean_'+args.split+'_'+str(epoch)+'/' + 'test.json', 'w', encoding='utf-8') as f:
                for idx in range(split_num*(train_num+dev_num), total_num):
                    one = data_tmp[idx]
                    tmp = json.dumps(one, ensure_ascii=False)
                    f.write(tmp + '\n')