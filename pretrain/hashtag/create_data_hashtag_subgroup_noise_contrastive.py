import argparse
import json
import random
from tqdm import tqdm,trange
import numpy as np
import re
import string
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file',default='../../bertopic/tweet_hash_clean_group_subgroup_100_0.json',type=str)
parser.add_argument('--num',default=100,type=int)
parser.add_argument('--name',default='tweet_hash_clean_group_raw',type=str)
parser.add_argument('--ran1', default=0.333, type=float)
parser.add_argument('--ran2', default=0.667, type=float)
parser.add_argument('--ran3', default=0.5, type=float)
parser.add_argument('--max_len', default=512, type=int)
parser.add_argument('--rep', default=5, type=int)
parser.add_argument('--balance', default=0, type=int)
parser.add_argument('--root', default='../../', type=str)
parser.add_argument('--sep', default=0, type=int)
parser.add_argument('--sim_idx',default='tweet_hash_clean_group_subgroup_sim.json',type=str)

args = parser.parse_args()

with open(args.root+'/contrastive/hash_seg10.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
hash_seg = {}
for line in lines:
    line = line.strip()
    hash_seg[line.split('\t')[0]] = line.split('\t')[1]


HASH = re.compile(r"#\S+")
def process(line):
    hash_tmp = HASH.findall(line)
    hash_tmp_clean = []
    for hash_one in hash_tmp:
        # hash_one = hash_one.lower()
        if len(hash_one) > 30:
            continue
        if hash_one[1].isalpha():
            if hash_one[-1] == '…':
                continue
            if len(hash_one) > 3 and hash_one[-3:] == '...':
                continue
            if hash_one[-1] in string.punctuation:
                hash_one = hash_one[:-1]
            hash_clean = re.findall('[a-zA-Z0-9]*', hash_one)
            hash_clean = '#' + ''.join(hash_clean)
            if hash_one == hash_clean:
                hash_tmp_clean.append(hash_one)

    return hash_tmp_clean

def read_data(args):
    hash_data = []
    with open(args.file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tmp = json.loads(line)
            hash_data.append(tmp)
    return hash_data

def cal_sim(hash_data):
    cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
    convert_idx = {}
    hash_emb = []
    for hash_idx in range(len(hash_data)):
        hash_emb_one = hash_data[hash_idx]['emb']
        if len(hash_emb_one) > 0:
            for idx_tmp in range(len(hash_emb_one)):
                hash_emb.append(hash_emb_one[idx_tmp])
                convert_idx[len(hash_emb)-1] = [hash_idx, idx_tmp]
        else:
            for idx_tmp in range(len(hash_data[hash_idx]['text'])):
                emb_tmp = list( np.zeros([len(hash_emb[0]),]) )
                hash_emb.append(emb_tmp)
                convert_idx[len(hash_emb) - 1] = [hash_idx, idx_tmp]
    hash_emb = torch.tensor(hash_emb)
    sim_idx = {}
    for hash_idx in range(len(hash_emb)):
        dis = cos_sim(hash_emb[hash_idx:hash_idx+1], hash_emb)
        val, best_idx = dis.topk(30)
        sim_tmp = []
        for tmp_idx in best_idx.cpu().numpy():
            if convert_idx[tmp_idx][0] != convert_idx[hash_idx][0]:
                sim_tmp.append(convert_idx[tmp_idx])
        idx_str = str(convert_idx[hash_idx][0])+':'+str(convert_idx[hash_idx][1])
        sim_idx[idx_str] = sim_tmp[:10]
    return sim_idx


def main(args, hash_data, sim_idx):
    hash_pair = []
    for hash_idx in trange(len(hash_data)):
        if args.balance == 0:
            epoch = args.rep
        else:
            epoch = min(args.rep, int( args.balance*args.rep*1.0/ sum([len(i) for i in hash_data[hash_idx]['text']]) ) )
        for tmp in range(epoch):
            hash_data_hash = hash_data[hash_idx]
            hash_data_text = hash_data_hash['text']
            for idx_tmp in range(len(hash_data_text)):
                hash_data_one = hash_data_text[idx_tmp]
                random.shuffle(hash_data_one)
                hash_data_one_noise = []
                hash_data_one_clean = []
                for data_tmp in hash_data_one:
                    data_tmp = data_tmp.strip()
                    data_tmp_clean = data_tmp.strip()
                    hash_tmp = process(data_tmp)
                    for hash_two in hash_tmp:
                        if hash_two.lower() == hash_data_hash['hashtag']:
                            data_tmp_clean = data_tmp_clean.replace(hash_two, '')
                        ran1 = np.random.random()
                        if ran1 < args.ran1:
                            data_tmp = data_tmp.replace(hash_two, '')
                        elif ran1 < args.ran2:
                            tmp2 = hash_seg.get(hash_two.lower())
                            if tmp2 is not None:
                                data_tmp = data_tmp.replace(hash_two, tmp2)
                            else:
                                data_tmp = data_tmp.replace(hash_two, hash_two[1:])

                    hash_data_one_noise.append(data_tmp)
                    hash_data_one_clean.append(data_tmp_clean)
                hash_data_group = ''
                for idx in range(len(hash_data_one_noise)):
                    if args.sep == 0:
                        hash_data_group += hash_data_one_noise[idx] + ' '
                    else:
                        hash_data_group += hash_data_one_noise[idx] + ' </s> '
                    if len(hash_data_group) > args.max_len * 0.95:
                        rand_one = random.random()
                        if rand_one > args.ran3:
                            text2 = ' '.join(random.sample(hash_data_one_clean, args.con_len))
                        else:
                            text2 = ' '.join(random.sample(hash_data_one_clean, 1))

                        hard_neg_idx = random.sample( sim_idx[str(hash_idx)+':'+str(idx_tmp)] , 1)[0]
                        hash_data_two = hash_data[hard_neg_idx[0]]['text'][hard_neg_idx[1]]
                        hash_data_two = random.sample(hash_data_two, min(len(hash_data_two),args.con_len))
                        hash_data_two_clean = []
                        for data_tmp in hash_data_two:
                            data_tmp = data_tmp.strip()
                            hash_tmp = process(data_tmp)
                            for hash_two in hash_tmp:
                                if hash_two.lower() == hash_data[hard_neg_idx[0]]['hashtag']:
                                    data_tmp = data_tmp.replace(hash_two, '')

                            hash_data_two_clean.append(data_tmp)
                        text3 = ' '.join(hash_data_two_clean)

                        hash_pair.append({'text1': hash_data_group, 'text2': text2, 'text3': text3, 'label': hash_idx})
                        hash_data_group = ''

if __name__ == '__main__':
    hash_data = read_data(args)
    if os.path.isfile(args.sim_idx):
        with open(args.sim_idx, 'r', encoding='utf-8') as f:
            sim_idx = json.load(f)
    else:
        sim_idx = cal_sim(hash_data)
        with open(args.sim_idx, 'w', encoding='utf-8') as f:
            tmp = json.dumps(sim_idx, ensure_ascii=False)
            f.write(tmp)
    main(args, hash_data, sim_idx)
