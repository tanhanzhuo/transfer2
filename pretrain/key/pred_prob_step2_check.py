import pickle

prob_map = []
for split in range(3):
    with open('/work/test/pretrain_hashtag/prob' + '_sep8_' + str(split) + '.pickle', 'rb') as handle:
        tmp = pickle.load(handle)
        # tmp = [list(tt) for tt in tmp]
        print(len(tmp))
        prob_map.extend(tmp)

# for split in range(4):
#     with open('/work/test/pretrain_hashtag/prob' + '_makeup_' + str(split) + '.pickle', 'rb') as handle:
#         tmp = pickle.load(handle)
#         tmp = [list(tt) for tt in tmp]
#         print(len(tmp))
#         prob_map.extend(tmp)



from transformers import AutoTokenizer
from modeling import RobertaForTokenClassification
import numpy as np
import paddle
from tqdm import tqdm,trange
from paddlenlp.data import Pad
import argparse

paddle.set_device("gpu:"+str(1))

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', normalization=True)
model = RobertaForTokenClassification.from_pretrained(
       '/work/test/pretrain_hashtag/keyphrase/model/twitter_hash_key', num_classes=2)
model.eval()
USER = tokenizer('@USER',add_special_tokens=False)['input_ids'][0]
URL = tokenizer('HTTPURL',add_special_tokens=False)['input_ids'][0]
START = tokenizer.bos_token_id
END = tokenizer.eos_token_id
SP = [USER, URL, START, END]

import datasets
train_dataset = datasets.load_from_disk('/work/test/pretrain_hashtag/twitter_ref_clean_simple/TrainData_line')["train"]
# train_dataset = datasets.load_from_disk('/work/test/pretrain_hashtag/txt_prob_test/TrainData_line')["train"]

def cal_prob(txt,PP=0.5):
    txt_token_all = tokenizer(txt,return_special_tokens_mask=True)
    txt_token = np.array(txt_token_all['input_ids'])
    if len(txt_token) > 128 or len(txt_token) < 4:
        return []
    prob = np.array([-1.0] * len(txt_token))
    for idx in range(len(txt_token)):
        if txt_token[idx] in SP:
            prob[idx] = 0

    txt_token_clean = txt_token[prob!=0]
    txt_token_sep=[[]]
    idx_cur = 0
    for idx in range(len(txt_token_clean)):
        one = tokenizer.decode([txt_token_clean[idx]])
        if one[-2:] != '@@':
            txt_token_sep[idx_cur].append(txt_token_clean[idx])
            idx_cur+=1
            if idx!=len(txt_token_clean)-1:
                txt_token_sep.append([])
        else:
            txt_token_sep[idx_cur].append(txt_token_clean[idx])
    txt_token_clean_len = [len(token) for token in txt_token_sep]

    txt_small_token_sep=[]
    txt_small_token_join=[]
    for token in txt_token_sep:
        word = tokenizer.decode(token)
        word_small = word.lower()
        word_small_token = tokenizer(word_small,add_special_tokens=False)['input_ids']
        txt_small_token_sep.append(word_small_token)
        txt_small_token_join.extend(word_small_token)
    txt_small_token_len = [len(token) for token in txt_small_token_sep]
    txt_small_token_join=[START] + txt_small_token_join + [END]
    with paddle.no_grad():
        logits = model(paddle.to_tensor([txt_small_token_join]), paddle.to_tensor([ [0]*len(txt_small_token_join) ]))
        preds=list(paddle.nn.functional.softmax(logits)[0,1:-1,1].cpu().numpy())
        # preds = [0]*(len(txt_small_token_join)-2)

    preds_reshape = []
    for idx in range(len(txt_small_token_len)):
        pop_len = txt_small_token_len[idx]
        preds_reshape.append([])
        for idx2 in range(pop_len):
            preds_reshape[idx].append(preds.pop(0))
    preds_reshape_max = [max(tmp) for tmp in preds_reshape]

    prob_tmp =[]
    for idx in range(len(txt_token_clean_len)):
        token_len = txt_token_clean_len[idx]
        for idx2 in range(token_len):
            prob_tmp.append(preds_reshape_max[idx])

    idx_tmp = 0
    for idx in range(len(prob)):
        if prob[idx] == 0:
            continue
        prob[idx] = prob_tmp[idx_tmp]
        idx_tmp+=1


    print( tokenizer.decode( np.array(txt_small_token_join)[ logits.argmax(axis=2).cpu().numpy()[0]==1 ] ) )
    print( tokenizer.decode( txt_token[prob>0.5] ) )
    # print( tokenizer.decode( [txt_token[np.argmax(prob)]] ) )
    # print( tokenizer.decode( txt_token[prob>PP] ) )
    
    txt_token_all['prob'] = prob
    return txt_token_all

idx = 8116#67591080*3+16897770*3+5512113
txt = tokenizer.decode(train_dataset[idx]['input_ids'])
print(txt)
cal_prob(txt)
prob = prob_map[idx]
# prob = np.array( train_dataset[idx]['prob'] )
print(tokenizer.decode(np.array(train_dataset[idx]['input_ids'])[prob>0.5]))

# prob_map = [list(p) for p in prob_map]
train_dataset = train_dataset.add_column("prob_map", prob_map)
train_dataset.save_to_disk('/work/test/pretrain_hashtag/TrainData_refine')