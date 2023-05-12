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
import faiss

parser = argparse.ArgumentParser()
parser.add_argument('--hash_file',default='./tweet_hash_clean_seg_one20/tweet_hash_clean_seg',type=str)
# parser.add_argument('--model',default='/work/SimCSE-main/result/thre1000_num1000/',type=str)
# parser.add_argument('--model_name',default='/work/SimCSE-main/result/thre100_num100_remove/1399999',type=str)
parser.add_argument('--model_name',default='../lmbff/contrastive_models/one/20_new/',type=str)
parser.add_argument("--max_seq_length", default=128, type=int)

parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument("--best", default=100, type=int)
parser.add_argument('--method',default='_seg_one20',type=str)
parser.add_argument("--split", default=50, type=int)#for gpu memory
parser.add_argument("--hashprocess", default='seg', type=str)#for gpu memory
parser.add_argument("--gpu", default=8, type=int)#for gpu memory
parser.add_argument("--num", default=1000, type=int)#for gpu memory

args = parser.parse_args()

time1 = time.time()
hash_samples = []
# hash_embs = np.array([]).reshape((-1,768))
hash_embs = []
for idx in trange(args.split):
    tmp = np.load(args.hash_file+'_'+str(idx)+'.npz',allow_pickle=True)
    # print(tmp['embs'].dtype)
    hash_samples.extend(tmp['samples'])
    # hash_embs = np.concatenate((hash_embs,tmp['embs']))
    hash_embs.extend(tmp['embs'])#,dtype=np.float16)
    tmp.close()

hash_embs = np.asarray(hash_embs)#,dtype=np.float16)
print(hash_embs.dtype)

time2 = time.time()
print('read time:{}'.format(time2-time1))

dim = len(hash_embs[0])

print('input dimension:')
print(hash_embs.shape)

# time1 = time.time()
# cpu_index = faiss.IndexFlatIP(dim)  # 构建索引index
# gpu_index = cpu_index#faiss.index_cpu_to_all_gpus(cpu_index)
# gpu_index.add(hash_embs)
# time2 = time.time()
# print('direct build time:{}'.format(time2-time1))
# k = 100  # 返回结果个数
# query = hash_embs[:args.num]  # 查询本身
#
# time1 = time.time()
# dis, ind = gpu_index.search(query, k)
# time2 = time.time()
# print('direct search time:{}'.format(time2-time1))
# print('shape of dis and idx')
# print(dis.shape)
# print(ind.shape)
#
# del cpu_index, gpu_index


train_s = time.time()
# quantizer = faiss.IndexFlatL2(dim)  # def the method of calculating distance (L2 distance, here)
# cpu_index = faiss.IndexIVFPQ(quantizer, dim, int(len(hash_embs)/100), 8, 8)  # construct the index
cpu_index = faiss.index_factory(dim, "IVF600000,PQ8")
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
gpu_index.train(hash_embs)                       # train the index on the data
train_e = time.time()
print('index build time: {}'.format(train_e - train_s))

cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, 'tweet_hash_clean_seg_one20_index.index')

time1 = time.time()
gpu_index.add(hash_embs)
time2 = time.time()
print('index add time:{}'.format(time2-time1))
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, 'tweet_hash_clean_seg_one20_index_add.index')


k = 100  # 返回结果个数
query = hash_embs[:args.num]  # 查询本身
gpu_index.nprobe = int(len(hash_embs)/1000)
time1 = time.time()
dis, ind = gpu_index.search(query, k)
time2 = time.time()
print('direct search time:{}'.format(time2-time1))
print('shape of dis and idx')
print(dis.shape)
print(ind.shape)

