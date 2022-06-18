import os
import time
from multiprocessing import Pool
from tqdm import tqdm, trange
import re
import json
import string
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pool',default=20,type=int)
parser.add_argument('--num',default=1000,type=int)


HASH = re.compile(r"#\S+")
filePath =  'twitter_hash_sample.txt'#'/work/data/twitter_hash.txt'#'twitter_hash_sample.txt'

def process(line):
    line = line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip()
    if len(line) < 10:
        return []
    hash_tmp = HASH.findall(line)
    hash_tmp_clean = []
    for hash_one in hash_tmp:
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
                hash_tmp_clean.append(hash_one)

    return hash_tmp_clean

def write_json(fileName,data):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            json.dump(one, f)
            f.write('\n')

if __name__ == "__main__":
    args = parser.parse_args()

    time1 = time.time()
    f_read = open(filePath, 'r', encoding='utf-8')
    pool = Pool(args.pool)
    process_data = pool.imap(process, f_read, args.num)

    hash_all = []
    for data in tqdm(process_data):
        hash_all.extend(data)
    pool.close()
    hash_dic = {}
    for hash_one in hash_all:
        if hash_one in hash_dic.keys():
            hash_dic[hash_one] += 1
        else:
            hash_dic[hash_one] = 1

    time2 = time.time()
    print(time2 - time1)

    for hash_one in list(hash_dic.keys()):
        if hash_dic[hash_one] < 1000:
            hash_dic.pop(hash_one)

    write_json('hash_his', hash_dic)


    # hash_dic = {}
    # with open(filePath, 'r', encoding='utf-8') as f:
    #     for line in tqdm(f):
    #         line = line.replace('[RT] ', '').replace('[USER] ', '').replace(' [HTTP]', '').strip()
    #         if len(line) < 10:
    #             continue
    #         hash_tmp = HASH.findall(line)
    #         hash_tmp_clean = []
    #         for hash_one in hash_tmp:
    #             hash_one = hash_one.lower()
    #             if len(hash_one) > 30:
    #                 continue
    #             if hash_one[1].isalpha():
    #                 if hash_one[-1] == '…':
    #                     continue
    #                 if len(hash_one) > 3 and hash_one[-3:] == '...':
    #                     continue
    #                 if hash_one[-1] in string.punctuation:
    #                     hash_one = hash_one[:-1]
    #                 hash_clean = re.findall('[a-z0-9]*', hash_one)
    #                 hash_clean = '#' + ''.join(hash_clean)
    #                 if hash_one == hash_clean:
    #                     hash_tmp_clean.append(hash_one)
    #
    #         for hash_one in hash_tmp_clean:
    #             if hash_one in hash_dic.keys():
    #                 hash_dic[hash_one] += 1
    #             else:
    #                 hash_dic[hash_one] = 1
    # time3 = time.time()
    # print(time3-time2)
    #
    # for hash_one in list(hash_dic.keys()):
    #     if hash_dic[hash_one] < 1000:
    #         hash_dic.pop(hash_one)
    #
    # write_json('hash_his4',hash_dic)