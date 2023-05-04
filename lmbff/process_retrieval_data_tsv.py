import csv
import json

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default='../finetune/data/', type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--dataset_path", default='../finetune/data/', type=str, required=False, help="dataset name")
parser.add_argument("--task_name", default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm', type=str, required=False, help="dataset name")
parser.add_argument('--method',default='_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst',type=str)
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
            tsv_writer.writerow([one['text'], one['label']])

def write_tsv_two(data, fileName):
    with open(fileName + '.tsv', 'w', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['sentence_A', 'sentence_B', 'label'])
        for one in data:
            tsv_writer.writerow([one['text'], one['text0'], one['label']])

if __name__ == "__main__":
    args = parser.parse_args()
    for task in args.task_name.split(','):
        for sp in ['train','dev','test']:
            data = []
            with open(args.dataset_path+task+'/'+sp + args.method + '.json', 'r', encoding='utf-8') as f:
                for line in f:
                    one = json.loads(line)
                    text_sp = one['text'].split(' \n ')
                    for idx in range(len(text_sp)-2):
                        one['text'+str(idx)] = text_sp[idx].strip()
                    one['text'] = text_sp[-2].strip()
                    data.append(one)
            write_tsv_one(data, args.dataset_path+task+'/'+sp)
            write_tsv_two(data, args.dataset_path + task + '/' + sp + args.method)

