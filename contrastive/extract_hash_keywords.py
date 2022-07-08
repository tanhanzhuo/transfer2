import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--thre',default=100,type=int)
parser.add_argument('--num',default=3000,type=int)
parser.add_argument('--model',default='nltk',type=str)
args = parser.parse_args()

import json
f=open('./selected_thre100_num500_index.json','r',encoding='utf-8')
hash_dic = json.load(f)
f.close()

from tqdm import tqdm, trange
import re
import string
import random
random.seed(0)
HASH = re.compile(r"#\S+")
filePath = '/work/data/twitter_hash.txt'#'twitter_hash_test.txt'#

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

hash_thre_list = list(hash_dic.values())
hash_data = {}
for hash_one in hash_thre_list:
    hash_data[hash_one] = set()
with open(filePath, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        hash_tmp_clean = process(line)
        for hash_one in hash_tmp_clean:
            tmp = hash_data.get(hash_one)

            if tmp is not None:
                hash_data[hash_one].add(
                    line.replace('[RT] ', '').replace('[USER]', '').replace('[HTTP]', '').replace('&amp','').strip()
                )

import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')

with open('stop1.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    line1 = [line.strip() for line in lines]
with open('stop2.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    line2 = [line.strip() for line in lines]
stop_words = list(stopwords.words('english'))
if args.model != 'nltk':
    stop_words = set(stop_words)
    for one in line1:
        stop_words.add(one)
    for one in line2:
        stop_words.add(one)
    stop_words = list(stop_words)
hash_word = {}

for hash_one in tqdm(hash_thre_list):
    tokens = {}

    data_tmp = list(hash_data[hash_one])
    idx_tmp = list(range(len(data_tmp)))
    if args.num < len(hash_data[hash_one]):
        random.shuffle(idx_tmp)
        idx_tmp = idx_tmp[:args.num]

    for idx_one in idx_tmp:
        one = data_tmp[idx_one].lower()
        tokens_one = word_tokenize(one)
        for word in tokens_one:
            word_clean = re.findall('[a-z0-9]*', word)
            word_clean = ''.join(word_clean)
            if word == word_clean and word not in stop_words:
                if word == 'rt':
                    continue
                tmp = tokens.get(word)
                if tmp is not None:
                    tokens[word]+=1
                else:
                    tokens[word] = 1
    for one in list(tokens.keys()):
        if tokens[one]<=5:
            tokens.pop(one)
    tokens_sort = dict(sorted(tokens.items(), key=lambda x: x[1],reverse=True))
    print(list(tokens_sort.keys())[:10])
    print(list(tokens_sort.values())[:10])
    hash_word[hash_one] = list(tokens_sort.keys())[:50]

if args.model == 'nltk':
    with open('./selected_thre'+str(args.thre)+'_num'+str(args.num) + '_word_nltk.json', 'w', encoding='utf-8') as f:
        json.dump(hash_word, f)
else:
    with open('./selected_thre'+str(args.thre)+'_num'+str(args.num) + '_word_all.json', 'w', encoding='utf-8') as f:
        json.dump(hash_word, f)

