import random
import numpy as np
import os

seed = 1
random.seed(seed)
np.random.seed(seed)
def convert_data(fileName, saveName):
    f = open(fileName, 'r')
    f_w = open(saveName, 'a')

    lines = f.readlines()
    for line in lines:
        line_split = line.split('\t')
        if len(line_split) == 3:
            f_w.write(line_split[1].strip() + '\t' + line_split[2].strip() + '\n')
        else:
            print(line)

    f.close()
    f_w.close()

convert_data('./data/sem-17/test_ori', './data/sem-17/test')

def convert_data_train(fileName, saveName):
    f = open(fileName, 'r')
    f_w_d = open(saveName + 'dev', 'a')
    f_w_t = open(saveName + 'train', 'a')

    lines = f.readlines()
    random.shuffle(lines)
    dev_size = int(0.1 * len(lines))
    for idx in range(dev_size):
        line = lines[idx].strip()
        line_split = line.split('\t')

        f_w_d.write(line_split[1].strip() + '\t' + line_split[2].strip() + '\n')

    for idx in range(dev_size, len(lines)):
        line = lines[idx].strip()
        line_split = line.split('\t')

        f_w_t.write(line_split[1].strip() + '\t' + line_split[2].strip() + '\n')

    f.close()
    f_w_d.close()
    f_w_t.close()

convert_data_train('./data/sem-17/train_ori', './data/sem-17/')

# def read_label(fileName):
#     with open(fileName, 'r') as f:
#         lines = f.readlines()
#         label_name = {}
#         for line in lines:
#             if line.split('\t')[0] in label_name.keys():
#                 label_name[line.split('\t')[0]] += 1
#             else:
#                 label_name[line.split('\t')[0]] = 1
#         f.close()
#     print(label_name)

# read_label('./data/sem-17/train')