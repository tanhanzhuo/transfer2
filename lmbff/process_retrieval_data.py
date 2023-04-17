import argparse
import json
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm,sem18-task1-affect,sem21-task7-humor', type=str, required=False, help="dataset name")
parser.add_argument('--method_hash',default='modelT100N100R_fileT100N100R_num10_top20_textfirst',type=str)
parser.add_argument('--top',default=1,type=int)
parser.add_argument('--name',default='',type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    for task in args.task_name.split(','):
        for sp in ['train','dev','test':]
            with open(args.dataset_path+task+'/'+sp, 'r', encoding='utf-8') as f:
                lines = f.readlines()
