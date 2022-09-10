import argparse
from scipy.spatial.distance import pdist, squareform,cosine
import numpy as np
import torch
from tqdm import tqdm,trange
parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='../contrastive_full/feature_modelT100N100M_fileT100N100S_num10',type=str)
parser.add_argument("--split", default=4, type=int)#for gpu memory
parser.add_argument("--piece", default=20, type=int)#for gpu memory
parser.add_argument("--thre", default=0.8, type=float)
parser.add_argument("--top", default=5, type=float)
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
hash_embs = torch.tensor(np.array(hash_embs)).cuda()
[num,dim]=hash_embs.shape
hash_embs = hash_embs.reshape(int(num/10),10,dim)
# hash_embs = torch.mean(hash_embs,1)

hash_merge=[]
cos_sim = torch.nn.CosineSimilarity(dim=-1)
# a=torch.tensor(numpy.random.random([5,3,17]))
# b=cos_sim(a,a.unsqueeze(1))
# b=cos_sim(a[0],a.unsqueeze(2))
# print(b[3,1,2],cos_sim(a[0][2],a2[3][1]))
file = open('hash_merge_'+args.save+'.txt', 'a', encoding='utf-8')
SP=20
BATCH = int( len(hash_tags)/SP )
for hash_idx in trange(len(hash_embs)):
    tmp_merge = ''+[hash_tags[hash_idx]]
    dis = []
    for one in range(SP+1):
        dis_tmp = cos_sim(hash_embs[hash_idx], hash_embs[one*BATCH:(one+1)*BATCH].unsqueeze(2)).mean(dim=[1,2])
        dis.extend(dis_tmp.cpu().numpy())
    dis = torch.tensor(dis)
    dis[hash_idx] = 0
    val,place = dis.topk(args.top)
    for idx_tmp in range(args.top):
        tmp_merge = tmp_merge + '\t' + hash_tags[place[idx_tmp].item()]
    tmp_merge += ' \n '
    file.write(tmp_merge)