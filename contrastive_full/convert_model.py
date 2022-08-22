import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModel,DataCollatorWithPadding,set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--bt',default='./bertweet/',type=str)
parser.add_argument('--ori',default='./1399999/',type=str)
parser.add_argument('--save',default='./1399999_new/',type=str)

parser.add_argument('--temp',default=0.05,type=float)
parser.add_argument('--pooler_type',default='cls',type=str)
parser.add_argument('--hard_negative_weight',default=0,type=float)
parser.add_argument('--do_mlm',default=False,type=bool)
parser.add_argument('--mlm_weight',default=0.1,type=float)
parser.add_argument('--mlp_only_train',default=False,type=bool)

args = parser.parse_args()

# model_sim = AutoModel.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
# model_sim.save_pretrained('simcse')
# model_sim = torch.load('./simcse/pytorch_model.bin')
# with open('simcse.txt','w',encoding='utf-8') as f:
#     for key in model_sim.keys():
#         f.write(key+'\n')

# model_bt = AutoModel.from_pretrained('vinai/bertweet-base')
# model_bt.save_pretrained('bertweet')
# model_bt = torch.load('./bertweet/pytorch_model.bin')

# with open('ori.txt','w',encoding='utf-8') as f:
#     for key in model_ori.keys():
#         f.write(key+'\n')
# with open('bertweet.txt','w',encoding='utf-8') as f:
#     for key in model_bt.keys():
#         f.write(key+'\n')

model_ori = torch.load(args.ori+'pytorch_model.bin')
from collections import OrderedDict
model_new =OrderedDict({})
for k, v in model_ori.items():
    k=k.replace('roberta.','').replace("mlp","pooler")
    model_new[k]=v
# model3 = OrderedDict(('pooler' if k == 'mlp' else k, v) for k, v in model2.items())
torch.save(model_new,args.save+'pytorch_model.bin')
import os
import shutil
shutil.copyfile(args.bt+'/config.json',args.save + "/config.json")

set_seed(0)
def compare(model_ori, model_convert):
    x = np.random.randint(
        1, model_ori.config.vocab_size, size=(4, 64))
    input_torch = torch.tensor(x, dtype=torch.int64)
    with torch.no_grad():
        out_ori = model_ori(input_ids=input_torch,
              output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        out_new = model_new(input_torch).pooler_output.detach()
        print(torch.sum(out_ori-out_new))


from models import RobertaForCL
config = AutoConfig.from_pretrained(args.ori)
model_ori = RobertaForCL.from_pretrained(
                args.ori,
                config=config,
                model_args=args
            )
model_ori.eval()

model_new = AutoModel.from_pretrained(args.save)
model_new.eval()

compare(model_ori,model_new)
