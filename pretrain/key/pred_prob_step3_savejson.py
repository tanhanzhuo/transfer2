import pickle
import emoji
from tqdm import tqdm,trange

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--SP", default=7, type=int)
args = parser.parse_args()

import datasets
train_dataset = datasets.load_from_disk('/work/test/pretrain_hashtag/twitter_ref_clean_simple/TrainData_line')["train"]
from bertweet_token import BertweetTokenizer
tokenizer = BertweetTokenizer.from_pretrained('vinai/bertweet-base')#, normalization=True)


SPLIT = 8
TOTAL = len(train_dataset)#270364321
BATCH = int(TOTAL /SPLIT)
IDX = []
for idx in range(SPLIT-1):
    IDX.append([BATCH*idx, BATCH*(idx+1)])
IDX.append([BATCH*(idx+1), TOTAL])

print(IDX[args.SP][0])
prob_map = []

# with open('/work/test/pretrain_hashtag/prob_test.pickle', 'rb') as handle:
#     tmp = pickle.load(handle)
#     tmp = [list(tt) for tt in tmp]
#     print(len(tmp))
#     prob_map.extend(tmp)

for split in [args.SP]:
    with open('/work/test/pretrain_hashtag/prob' + '_sep8_' + str(split) + '.pickle', 'rb') as handle:
        tmp = pickle.load(handle)
        tmp = [list(tt) for tt in tmp]
        print(len(tmp))
        prob_map.extend(tmp)

import json
with open('/work/test/pretrain_hashtag/txt_prob_200m_' + str(args.SP) + '.json', 'w') as f:
    for idx in trange( len(prob_map) ):
        one = train_dataset[IDX[args.SP][0] + idx]['input_ids'][1:-1]
        if len(one) <= 3:
                continue
        txt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(one))
        txt = txt.replace('<unk>','💙')

        # split_tokens = []
        # for token in txt.split(' '):
        #     split_tokens.extend([t for t in tokenizer.bpe(token).split(" ")])
        # two = tokenizer.convert_tokens_to_ids(split_tokens)
        two = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
        if one != two:
            continue

        prob = prob_map[idx]
        out_dic = {'text':txt,'prob':prob}
        out_dic = json.dumps(out_dic, ensure_ascii=False)
        f.write(out_dic+'\n')
