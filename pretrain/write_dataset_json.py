import datasets
import jsonlines
from tqdm import tqdm, trange
train_file = '/work/test/hf/test-mlm-ernie-ref/TrainData_line'
# train_file = '/work/test/hf/collator/hf/sep3/TrainData_line'
# train_file = '/work/test/hf/collator/hf/sep/TrainData_line'
tokenized_datasets = datasets.load_from_disk(train_file)
train_dataset = tokenized_datasets["train"]
# sum_len = 0
# SCALE = 100
# LEN = len(train_dataset)
# for i in trange(int(LEN/SCALE)):
#     tmp = train_dataset[i*SCALE:(i+1)*SCALE]['input_ids']
#     sum_len += sum(map(len,tmp))
#     if (i+1) % 1000 == 0:
#         print(sum_len*1.0/i/SCALE)
# print(sum_len*1.0/(i-1)/SCALE)


train_dataset = train_dataset.shuffle()

# with open(train_file + '.json', 'w') as f:
#     for one in tqdm(train_dataset):
#         one.pop('attention_mask')
#         json.dump(one, f)
#         f.write('\n')

# with jsonlines.open(train_file + '.jsonl', mode = 'w') as f:
#     for one in tqdm(train_dataset):
#         one.pop('attention_mask')
#         f.write(one)


from multiprocessing import Pool

def write_data(train_file, idx_file, train_dataset):
    NUM = 8
    NUM_SP = int(len(train_dataset) / NUM)
    with jsonlines.open(train_file + '_sep_' + str(idx_file) + '.jsonl', mode = 'w') as f:
        for idx_data in trange(NUM_SP):
            one = train_dataset[idx_file*NUM + idx_data]
            one.pop('attention_mask')
            f.write(one)
NUM = 8
pool = Pool(8)
for idx_file in range(NUM):
    pool.apply_async(write_data, args=(train_file, idx_file, train_dataset))
pool.close()
pool.join()