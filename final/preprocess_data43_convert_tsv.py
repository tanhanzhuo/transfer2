import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method',default='_seg_500_one20_top100_sp',type=str)
#_simcse_fileT100N100S_num10_cluster_top20_textfirst
parser.add_argument('--top',default=1,type=int)
parser.add_argument('--name',default='',type=str)

def write_json(data, fileName):
    with open(fileName + '.json', 'w', encoding='utf-8') as f:
        for one in data:
            tmp = json.dumps(one, ensure_ascii=False)
            f.write(tmp+'\n')

def write_tsv_one(data, fileName):
    with open(fileName + '.tsv', 'w', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['sentence', 'label'])
        for one in data:
            if len(one['text']) > 500:
                one['text'] = one['text'][:500]
            tsv_writer.writerow([one['text'], one['labels']])

def write_tsv_two(data, fileName):
    with open(fileName + '.tsv', 'w', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['sentence_A', 'sentence_B', 'label'])
        for one in data:
            if len(one['text']) > 500:
                one['text'] = one['text'][:500]
            if len(one['text0']) > 500:
                one['text0'] = one['text0'][:500]
            tsv_writer.writerow([one['text'], one['text0'], one['labels']])

if __name__ == "__main__":
    args = parser.parse_args()
    for task in args.task_name.split(','):
        for sp in ['train','dev','test']:
            data = []
            with open(args.dataset_path+task+'/'+sp + args.method + '.json', 'r', encoding='utf-8') as f:
                for line in f:
                    one = json.loads(line)
                    data.append(one)
            write_tsv_one(data, args.dataset_path+task+'/'+sp)
            write_tsv_two(data, args.dataset_path + task + '/' + sp + args.method)

