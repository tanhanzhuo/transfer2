import json
import datasets
import argparse
import torch
import numpy as np
from tqdm import tqdm,trange
import emoji
from scipy.spatial.distance import pdist, squareform
import time
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='./tweet_hash_clean_seg_one20/tweet_hash_clean_seg',type=str)
args = parser.parse_args()

time1 = time.time()
hash_samples = []
# hash_embs = np.array([]).reshape((-1,768))
hash_embs = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    embs = normalize(tmp['embs'])
    samples = tmp['samples']
    tmp.close()
    np.savez(args.hash_file+'_'+str(idx)+'.npz', embs=embs.astype(np.float16),
             samples=samples)

