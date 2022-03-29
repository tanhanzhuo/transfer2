import random
from transformers import AutoTokenizer
# import pycld2 as cld2
import fasttext
from tqdm import tqdm, trange
import numpy as np

TOTAL = 200000000
TRAIN = 5000000
TEST = 100000
model = fasttext.load_model('/work/test/pretrain_hashtag/keyphrase/lid.176.bin')

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
hash_dic = {}
hash_bad = []

def check_hash(line):
    # line = line.replace('</s>', '[SEP]')
    # line = line.replace('<mask>', '[MASK]')
    line = line.replace('@USER', '')###################VIP
    line = line.replace('HTTPURL', '')###################VIP

    input_ids = tokenizer(line)['input_ids']
    if len(input_ids) > 128:
        return False,0,0
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    # eos_token_id = tokenizer.sep_token_id
    LEN = len(input_ids)
    mask_idx = -1
    for token_id in range(LEN):
        if input_ids[token_id] == mask_token_id:
            mask_idx = token_id
            break
    eos_idx = -1
    for token_id in range(LEN-2, -1, -1):
        if input_ids[token_id] == eos_token_id:
            eos_idx = token_id
            break
    if mask_idx == -1 or eos_idx == -1:############wrong case
        return False,0,0
    if mask_idx<=1 or eos_idx-mask_idx<=2:#############at start or end case
        return False,0,0

    hash_sep = tokenizer.decode( input_ids[eos_idx+1:LEN-1] )##############exclude strange phrases
    # if len( hash_sep.replace(' ','') ) <= 5 or len(hash_sep.split()) <= 2: #################do not split words like axure, lol, cc20
    #     return True,hash_sep.replace(' ','').lower(),line.lower()

    hash_split = hash_sep.split()
    if len(hash_split) == 1: #################single word
        return True,hash_sep.lower(),line.lower()

    hash_len = [len(i) for i in hash_split]
    if np.mean(hash_len) < 3:####################exclude 'raj at ki in ni'
        return False,0,0
    if len(hash_split) > 5:####################exclude too long case
        return False,0,0

    results = model.predict(hash_sep, k=1)[0]
    # isReliable, textBytesFound, details = cld2.detect(hash_sep)
    # if isReliable and details[0][0] == 'ENGLISH':
    if '__label__en' in results:################language check
        return True,hash_sep.replace(' ','').lower(),line.lower()
    else:
        return False,0,0
    
    

with open('/work/test/pretrain_hashtag/twitter_hash_mask_bt.txt', 'r') as f:
# with open('/work/test/pretrain_hashtag/keyphrase/twitter_hash_key/test.txt', 'r') as f:
    lines = f.readlines()
    length = len(lines)
    lines_select = random.sample(lines, TOTAL)
    # lines_train = lines_select[:TRAIN]
    # lines_test = lines_select[TRAIN:]

progress_bar = tqdm(range(TRAIN))
count = 0
with open('/work/test/pretrain_hashtag/keyphrase/twitter_hash_key/train3m.txt','w') as f:
    for idx in range(len(lines_select)):
    # for line in lines_train:
        line = lines_select[idx]
        log, hash_one, line = check_hash(line)
        if log:
            if hash_one in hash_dic:
                if hash_dic[hash_one] >= 5:
                    continue
                else:
                    hash_dic[hash_one]+=1
                    f.write(line)
                    count += 1
                    progress_bar.update(1)
                    
            else:
                hash_dic[hash_one]=1
                f.write(line)
                count += 1
                progress_bar.update(1)
                
        if count > TRAIN:
            break
        if count %500000 ==0:
            print(idx)

progress_bar = tqdm(range(TEST))
count = 0
with open('/work/test/pretrain_hashtag/keyphrase/twitter_hash_key/test3m.txt','w') as f:
    for idx2 in range(idx, len(lines_select)):
        line = lines_select[idx2]
        log, hash_one, line = check_hash(line)
        if log:
            if hash_one not in hash_dic:
                f.write(line)
                count += 1
                progress_bar.update(1)
        if count > TEST:
            break
    import sys
    sys.exit()
print('done!!')
# with open('/work/test/pretrain_hashtag/twitter_hash_mask_bt.txt', 'r') as f:
#     lines = f.readlines()
#     length = len(lines)
#     lines_select = random.sample(lines, TRAIN+TEST)
#     lines_train = lines_select[:TRAIN]
#     lines_test = lines_select[TRAIN:]
# with open('/work/test/pretrain_hashtag/keyphrase/twitter_hash_key_train.txt','w') as f:
#     for line in lines_train:
#         f.write(line)
# with open('/work/test/pretrain_hashtag/keyphrase/twitter_hash_key_test.txt','w') as f:
#     for line in lines_test:
#         f.write(line)