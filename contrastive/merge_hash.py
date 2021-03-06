import argparse
from scipy.spatial.distance import pdist, squareform,cosine
import numpy as np
import torch
from tqdm import tqdm,trange
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='feature_modelT100N100S_fileT100S_num1',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory
parser.add_argument("--thre", default=0.9, type=float)
parser.add_argument('--save',default='cluster',type=str)
args = parser.parse_args()

hash_embs = []
hash_tags = []
for idx in range(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    # hash_embs.extend(tmp['center_embs'])
    hash_embs.extend(tmp['center_embs'])
    hash_tags.extend(tmp['center_hash'])
    tmp.close()
hash_embs = torch.tensor(hash_embs).cuda()

NUM=20
THRE=args.thre
hash_merge=[]

import torch
cos_sim = torch.nn.CosineSimilarity(dim=1)
for hash_idx in trange(len(hash_embs)):
    merge_idx = -1
    for tmp_idx in range(len(hash_merge)):
        if hash_tags[hash_idx] in hash_merge[tmp_idx]:
            merge_idx = tmp_idx
            break
    if merge_idx == -1:
        merge_idx = len(hash_merge)
        hash_merge.append(set([hash_tags[hash_idx]]))

    dis = cos_sim(hash_embs[hash_idx], hash_embs)
    dis[hash_idx] = 0
    val,place = dis.topk(NUM)
    for idx_tmp in range(NUM):
        if val[idx_tmp].item() >= THRE:
            hash_merge[merge_idx].add(hash_tags[place[idx_tmp].item()])

print(len(hash_merge))
import pickle
with open('hash_merge_'+args.save+'.pickle', 'wb') as f:
    pickle.dump(hash_merge, f)

# with open('hash_merge.pickle', 'rb') as f:
#     hash_merge = pickle.load(f)
#         print(hash_tags[idx],hash_tags[place[idx_tmp].item()],val[idx_tmp].item())