import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,sem-18,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm,sem18-task1-affect,sem21-task7-humor', type=str, required=False, help="dataset name")
args = parser.parse_args()

def read_data(fileName):
    with open(fileName, 'r', encoding='utf-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            data.append({'labels': line.split('\t')[0], 'text': line.split('\t')[1]})
    return data

TT=0
for task in args.task_name.split(','):
    total_num = 0
    for fileName in ['train', 'dev', 'test']:
        train_dataset = read_data(args.dataset_path + task + '/' + fileName)
        total_num += len(train_dataset)
    TT+=total_num
    print("task:{}, num:{}".format(task,total_num))
print(TT)

stance,hate,sem-18,sem22-task6-sarcasm
sem-17
imp-hate
sem19-task5-hate,sem19-task6-offen

CUDA_VISIBLE_DEVICES=0 python contrastive_process_data_cosine_singleGPU.py --hash_file feature_modelT100N100S_fileT100S_num10 --model /work/SimCSE-main/result/thre100_num100_seg/599999/ --task_name stance,hate,sem-18,sem22-task6-sarcasm --method _modelT100N100S_fileT100S
CUDA_VISIBLE_DEVICES=1 python contrastive_process_data_cosine_singleGPU.py --hash_file feature_modelT100N100S_fileT100S_num10 --model /work/SimCSE-main/result/thre100_num100_seg/599999/ --task_name sem-17 --method _modelT100N100S_fileT100S
CUDA_VISIBLE_DEVICES=2 python contrastive_process_data_cosine_singleGPU.py --hash_file feature_modelT100N100S_fileT100S_num10 --model /work/SimCSE-main/result/thre100_num100_seg/599999/ --task_name imp-hate --method _modelT100N100S_fileT100S
CUDA_VISIBLE_DEVICES=3 python contrastive_process_data_cosine_singleGPU.py --hash_file feature_modelT100N100S_fileT100S_num10 --model /work/SimCSE-main/result/thre100_num100_seg/599999/ --task_name sem19-task5-hate,sem19-task6-offen --method _modelT100N100S_fileT100S

