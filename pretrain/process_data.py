import os
from tqdm import tqdm, trange
import re
import numpy as np
from nltk.tokenize import wordpunct_tokenize
from emoji import demojize
import wordninja
from multiprocessing import Pool
import time

HASH = re.compile(r"#\S+")
filePath = '/work/test/twitter_hash.txt'
# filePath = '/work/test/sep3/2013_sep0.txt'

def process(line):
    line = line.strip()
    if not line:
        return ''
    line = demojize(line)
    hash_tmp = HASH.findall(line)
    
    for hash_one in hash_tmp:
        hash_sep = wordpunct_tokenize(hash_one)
        if hash_sep[0] == '#':
            line_tmp = line.replace(hash_one,'[MASK]')
            for hash_two in hash_tmp:
                line_tmp = line_tmp.replace(hash_two,'')
                
            line_tmp = line_tmp + ' [SEP] '
            hash_words = wordninja.split(hash_sep[1])
            if len(hash_words) > 0:
                for hash_word in hash_words:
                    line_tmp = line_tmp + hash_word +' '
                return line_tmp + '\n'
    return ''
if "__main__" == __name__:
    t1 = time.time()
    f_read = open(filePath, 'r')
    pool = Pool(20)
    process_data = pool.imap(process, f_read, 256)
    # print(process_data)

    with open('twitter_hash_mask.txt', 'w') as f_write:
        for data in tqdm(process_data):
            for sen in data:
                if sen:
                    f_write.write(sen)
    t2 = time.time()
    print(t2-t1)

    # t1 = time.time()
    # f_read = open(filePath, 'r')
    # pool = Pool(10)
    # process_data = pool.map(process, f_read)
    # # print(process_data)

    # with open('twitter_hash_mask2.txt', 'w') as f_write:
    #     for data in process_data:
    #         for sen in data:
    #             f_write.write(sen)
    # t2 = time.time()
    # print(t2-t1)

    # data = []
    # f_read = open(filePath, 'r')
    # with open('twitter_hash_mask2.txt', 'w') as f_write:
    #     for line in f_read:
    #         f_write.write(process(line))
    # t3 = time.time()
    # print(t3-t2)
    # f_read.close()