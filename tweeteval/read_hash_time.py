import json
import datasets
import numpy as np
from sklearn import metrics
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", default='../contrastive_full/feature_modelT100N100M_fileT100N100S_num10_cluster', type=str, required=False)
parser.add_argument("--N", default=10000, type=int, required=False)

args = parser.parse_args()

hash_embs = []
hash_tags = []
for idx in range(4):
    tmp = np.load(args.file+'_'+str(idx)+'.npz',allow_pickle=True)
    hash_embs.extend(tmp['center_embs'])
    tmp.close()

time1 = time.time()
for idx in range(args.N):
    d=metrics.pairwise.cosine_similarity([hash_embs[idx]],hash_embs)
time2 = time.time()
print((time2-time1)/args.N)