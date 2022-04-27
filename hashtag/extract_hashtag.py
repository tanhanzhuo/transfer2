import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hash', default=100,type=int)
parser.add_argument('--num', default=1000,type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    SELECT_HASH = args.hash
    SELECT_NUM = args.num

    with open('hash_his.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
    hash_dic = {}
    for line in lines:
        line = line.strip()
        hashtag = line.split('\t')[0]
        num = int(line.split('\t')[1])
        if num > 10000:
            hash_dic[hashtag] = num
    hash_dic_sort = dict(sorted(hash_dic.items(), key=lambda x: x[1], reverse=True))
    SELECT = list(hash_dic_sort.keys())[:SELECT_HASH]
    hash_dic = {}
    for one in SELECT:
        hash_dic[one] = 0


    import os
    from tqdm import tqdm, trange
    import random
    import re
    import json
    import string
    HASH = re.compile(r"#\S+")
    filePath = '/work/data/twitter_hash.txt'
    with open(filePath, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)

    data_hash = []
    for line in tqdm(lines):
        if not line:
            continue
        hash_tmp = HASH.findall(line)
        hash_one = hash_tmp[0]

        hash_one = hash_one.lower()
        if len(hash_one) > 30:
            continue
        if hash_one[1].isalpha():
            if hash_one[-1] == '…':
                continue
            if len(hash_one) > 3 and hash_one[-3:] == '...':
                continue
            if hash_one[-1] in string.punctuation:
                hash_one = hash_one[:-1]
            hash_clean = re.findall('[a-z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                if hash_one in SELECT:
                    if hash_dic[hash_one] < SELECT_NUM:
                        hash_dic[hash_one] += 1
                        data_line = line
                        for tmp in hash_tmp:
                            data_line = data_line.replace(tmp, '')
                        data_hash.append(hash_one + '\t' + data_line)

    with open('data_hash_' + str(SELECT_HASH) + '_' + str(SELECT_NUM) +'.txt', 'w') as f:
        for data in data_hash:
            f.write(data)
