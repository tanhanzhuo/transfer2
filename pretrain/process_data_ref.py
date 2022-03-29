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
filePath = '/work/test/twitter_ref.txt'
# filePath = '/work/test/sep3/2013_sep0.txt'

def process(line):
    
    line = line.replace('[RT]', '')
    line = line.strip()
    if not line:
        return ''
    line = line.replace('[USER]', '@USER')###################VIP
    line = line.replace('[HTTP]', 'HTTPURL')###################VIP
  
    hash_tmp = HASH.findall(line)
    line = demojize(line)

    for hash_one in hash_tmp:
        hash_sep = wordpunct_tokenize(hash_one)
        if hash_sep[0] == '#':
            hash_words = wordninja.split(hash_sep[1])
            # line_tmp = line.replace(hash_one,'[MASK]')
            if len(hash_words) > 0:
                line_tmp = line.replace(hash_one, ' '.join(hash_words))
                for hash_two in hash_tmp:
                    line_tmp = line_tmp.replace(hash_two,'')
                return line_tmp + '\n'

    return ''

def process_simple(line):
    
    line = line.replace('[RT]', '')
    line = line.strip()
    if not line:
        return ''
    
    line_clean = line.replace('[USER]', '').replace('[HTTP]', '')
    line_clean = re.sub("[^A-Za-z]", "", line_clean)
    if len(line_clean) < 10:
        return ''


    line = line.replace('[USER]', '@USER')###################VIP
    line = line.replace('[HTTP]', 'HTTPURL')###################VIP
    line = line.replace('...', '…')

    if line[-1] == '…':
        line_sp = line.split()
        line = ' '.join(line_sp[:-1])
    if len(line.replace(" HTTPURL", "")) <= 1:
        return ''
    if line.replace(" HTTPURL", "")[-1] == '…':
        line_tmp = line.replace(" HTTPURL", "")
        line_sp = line_tmp.split()
        line = ' '.join(line_sp[:-1]) + ' HTTPURL'

    # line = demojize(line)

    return line+'\n'

if "__main__" == __name__:
    t1 = time.time()
    f_read = open(filePath, 'r')
    pool = Pool(20)
    process_data = pool.imap(process_simple, f_read, 256)
    # print(process_data)

    with open('/work/test/pretrain_hashtag/twitter_ref_clean_simple.txt', 'w') as f_write:
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