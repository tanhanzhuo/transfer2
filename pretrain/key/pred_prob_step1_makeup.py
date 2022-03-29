# import datasets
# from tqdm import tqdm, trange
# train_dataset = datasets.load_from_disk('/work/test/pretrain_hashtag/twitter_ref_clean_simple/TrainData_line')["train"]
# SPLIT = 4
# TOTAL = 270364321
# BATCH = int(TOTAL /SPLIT)
# IDX = []
# for idx in range(SPLIT-1):
#     IDX.append([BATCH*idx, BATCH*(idx+1)])
# IDX.append([BATCH*(idx+1), TOTAL])


# IDX[3][1] = 266150749

# new_data = []
# for idx in trange(IDX[3][0], IDX[3][1]):
#     new_data.append( train_dataset[idx]['input_ids'] )

# import pickle
# with open('/work/test/pretrain_hashtag/makeup.pickle', 'wb') as handle:
#     pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


from transformers import AutoTokenizer
from modeling import RobertaForTokenClassification
import numpy as np
import paddle
from tqdm import tqdm,trange
from paddlenlp.data import Pad
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--NUM_SPLIT", default=4, type=int)
parser.add_argument("--CUR_SPLIT", default=0, type=int)
parser.add_argument("--CUR_GPU", default=1, type=int)
args = parser.parse_args()


import pickle
with open('/work/test/pretrain_hashtag/makeup.pickle', 'rb') as handle:
    train_dataset = pickle.load(handle)

paddle.set_device("gpu:"+str(args.CUR_GPU))
SPLIT = args.NUM_SPLIT
TOTAL = len(train_dataset)#270364321
BATCH = int(TOTAL /SPLIT)
IDX = []
for idx in range(SPLIT-1):
    IDX.append([BATCH*idx, BATCH*(idx+1)])
IDX.append([BATCH*(idx+1), TOTAL])

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
model = RobertaForTokenClassification.from_pretrained(
       '/work/test/pretrain_hashtag/keyphrase/model/twitter_hash_key', num_classes=2)
model.eval()
USER = tokenizer('@USER',add_special_tokens=False)['input_ids'][0]
URL = tokenizer('HTTPURL',add_special_tokens=False)['input_ids'][0]
START = tokenizer.bos_token_id
END = tokenizer.eos_token_id
SP = [USER, URL, START, END]

# import datasets
# train_dataset = datasets.load_from_disk('/work/test/pretrain_hashtag/TrainData_line')["train"]
BS = 256
pad1 = Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32')
pad2 = Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32')


# progress_bar = tqdm(range(BATCH))
# input_ids = train_dataset['input_ids']
prob_write = []
# IDX[args.CUR_SPLIT][0], IDX[args.CUR_SPLIT][1]
for idx_train in trange( int(BATCH/BS)+1 ):
    start_cur = IDX[args.CUR_SPLIT][0] + idx_train*BS
    end_cur = min( IDX[args.CUR_SPLIT][0] + (idx_train+1)*BS, IDX[args.CUR_SPLIT][1] )
    if idx_train == 434:
        aa =1
    batch = train_dataset[start_cur:end_cur]
    input_ids_batch = batch###['input_ids']

    txt_small_token_join_all = []
    txt_small_token_len_all = []
    txt_token_clean_len_all = []
    prob_all = []
    txt_type_id_all = []
    for txt_token in input_ids_batch:
        txt_token = np.array(txt_token)
        # if len(txt_token) > 128 or len(txt_token) < 4:
        #     return []
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

        txt_small_token_join_all.append(txt_small_token_join)
        txt_small_token_len_all.append(txt_small_token_len)
        txt_token_clean_len_all.append(txt_token_clean_len)
        prob_all.append(prob)
        txt_type_id_all.append( [0]*len(txt_small_token_join) )
    
    with paddle.no_grad():
        logits = model(paddle.to_tensor(pad1(txt_small_token_join_all)), paddle.to_tensor(pad2(txt_type_id_all)))
        preds_all=list(paddle.nn.functional.softmax(logits)[:,1:-1,1].cpu().numpy())
        # preds = [0]*(len(txt_small_token_join)-2)
    # preds_all = pad2(txt_type_id_all)
    for idx_bs in range(len(input_ids_batch)):
        txt_small_token_join=txt_small_token_join_all[idx_bs]
        txt_small_token_len=txt_small_token_len_all[idx_bs]
        txt_token_clean_len=txt_token_clean_len_all[idx_bs]
        prob=prob_all[idx_bs]
        if np.sum(np.abs(prob)) != 0:
            preds = list(preds_all[idx_bs])
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
        prob_write.append(prob)
        # print( tokenizer.decode( np.array(txt_small_token_join)[ logits[idx_bs,:len(txt_small_token_join)].argmax(axis=-1).cpu().numpy()==1 ] ) )
        # print( tokenizer.decode( np.array(input_ids_batch[idx_bs])[prob>0.5] ) )
    # progress_bar.update(BS)

        # print( tokenizer.decode( np.array(txt_small_token_join)[ logits.argmax(axis=2).cpu().numpy()[0]==1 ] ) )
        # print( tokenizer.decode( txt_token[prob>0.5] ) )
        # print( tokenizer.decode( [txt_token[np.argmax(prob)]] ) )
        # print( tokenizer.decode( txt_token[prob>PP] ) )
        
        # txt_token_all['prob'] = prob
        # return txt_token_all

import pickle
with open('/work/test/pretrain_hashtag/prob' + '_makeup_' + str(args.CUR_SPLIT) + '.pickle', 'wb') as handle:
    pickle.dump(prob_write, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('/work/test/pretrain_hashtag/prob' + '_sep_' + str(args.CUR_SPLIT) + '.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# import jsonlines
# with jsonlines.open('/work/test/pretrain_hashtag/prob' + '_sep_' + str(args.CUR_SPLIT) + '.jsonl', mode = 'w') as f:
#     for cur_data in prob_write:
#         f.write({'prob':list(cur_data)})
