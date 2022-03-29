import datasets
import numpy as np
import time
train_dataset = datasets.load_from_disk('/work/test/hf/collator/test-mlm-ernie-sep/TrainData_line')["train"]

prob_map = []
for idx in range(len(train_dataset)):
    # prob_map.append(np.array([0.1,0.6,0.8,0.4]))
    prob_map.append([0.1*idx,0.6*idx,0.8*idx,0.4*idx])
t1 = time.time()
train_dataset = train_dataset.add_column("prob_map", prob_map)
train_dataset.save_to_disk('/work/test/pretrain_hashtag/tt/TrainData_refine')
t2 = time.time()

print(t2-t1)