import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import time
import copy
# from accelerate import Accelerator
# accelerate = Accelerator()
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='../contrastive_full/features_simcse/twitter_hash_join_thre100_num100',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
parser.add_argument("--split", default=50, type=int)#for gpu memory
parser.add_argument("--N", default=100, type=int)#for gpu memory
#simcse

args = parser.parse_args()

hash_samples = []
hash_embs = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_samples.append(tmp['samples'])
    # hash_embs.extend(tmp['center_embs'])
    if idx < args.split / 2:
        hash_embs.append(torch.tensor(tmp['embs'], dtype=torch.float16).cuda(0))
    else:
        hash_embs.append(torch.tensor(tmp['embs'], dtype=torch.float16).cuda(1))
    tmp.close()

# hash_embs= torch.tensor(np.array(hash_embs))
cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()

with torch.no_grad():
    for idx in trange(args.N):
        outputs = np.array([np.random.rand(1,768)])
        for sp in range(args.split):
            if sp < args.split / 2:
                outputs = torch.tensor(outputs, dtype=torch.float16).cuda(0)
            else:
                outputs = torch.tensor(outputs, dtype=torch.float16).cuda(1)
            dis = cos_sim(outputs, hash_embs[sp])
            # dis = dis.view(-1,args.num_samples).sum(dim=-1)##################################hash each
            # best_idx = np.argpartition(np.array(dis), -args.best)[-args.best:]
            val,best_idx = dis.topk(1)