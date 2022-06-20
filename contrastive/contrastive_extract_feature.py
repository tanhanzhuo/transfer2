import json
import datasets
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file',default=None,type=str)
parser.add_argument('--model',default=None,type=str)
parser.add_argument('--save',default=None,type=str)
#simcse
parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

args = parser.parse_args()

# data = []
# with open(args.file, 'r', encoding='utf-8') as f:
#     if args.file.split('.')[-1] == 'txt':
#         data = f.readlines()
#     elif args.file.split('.')[-1] == 'json':
#         for line in f:
#             tmp = json.loads(line)
#             data.append(tmp['text1'])
#             data.append(tmp['text2'])
#     else:
#         print('error!!!!!!!!!!!!!!!')
raw_datasets = datasets.load_dataset('text', data_files=args.file)


from transformers import BertweetTokenizer, AutoConfig
from models import RobertaForCL

config = AutoConfig.from_pretrained(args.model)
model = RobertaForCL.from_pretrained(
    args.model,
    config=config,
    model_args=args
)


