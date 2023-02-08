import json
import numpy as np
from collections import OrderedDict
# od = collections.OrderedDict(sorted(d.items()))
# dict(sorted(people.items(), key=lambda item: item[1]))
f=open('./hash_his_clean.json','r',encoding='utf-8')
hash_dic = json.load(f)
f.close()

print(len(hash_dic.keys()))
print(sum([i for i in hash_dic.values()]))

hash_his = {}
N = 200
M = 5001
for i in range(0,M,N):
    hash_his[i] = 0
hash_his[M] = 0
for value in hash_dic.values():
    idx = int(value/N)
    if value > M:
        hash_his[M] += 1
    else:
        hash_his[idx*N] += 1

for key in hash_his.keys():
    print('{},{}'.format(key,hash_his[key]))
