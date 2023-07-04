from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import argparse
from transformers import RobertaForSequenceClassification
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import json

class RobertaForMulti(RobertaForSequenceClassification):
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        num_old = self.roberta.config.max_position_embeddings
        if num_old >= new_num_position_embeddings:
            return
        self.roberta.config.max_position_embeddings = new_num_position_embeddings
        # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
        new_position = nn.Embedding(self.roberta.config.max_position_embeddings, self.roberta.config.hidden_size)
        new_position.to(self.roberta.embeddings.position_embeddings.weight.device,
                        dtype=self.roberta.embeddings.position_embeddings.weight.dtype)
        # self._init_weights(new_position)
        new_position.weight.data[:num_old, :] = self.roberta.embeddings.position_embeddings.weight.data[:num_old, :]
        self.roberta.embeddings.position_embeddings = new_position

        self.roberta.embeddings.register_buffer("position_ids",
                                                torch.arange(self.roberta.config.max_position_embeddings).expand(
                                                    (1, -1)))
        self.roberta.embeddings.register_buffer(
            "token_type_ids", torch.zeros([1, self.roberta.config.max_position_embeddings], dtype=torch.long),
            persistent=False
        )

CONVERT = {
    'eval-emotion':{'0':0,'1':1,'2':2,'3':3},
    'eval-hate':{'0':0,'1':1},
    'eval-irony':{'0':0,'1':1},
    'eval-offensive':{'0':0,'1':1},
    'eval-stance': {'0': 0, '1': 1, '2': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task_name",
    # default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm',
    default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm',
    type=str,
    required=False,
    help="The name of the task to train selected in the list: ")

parser.add_argument(
    "--seed",
    default='0,1,2,3,4,5,6,7,8,9',
    type=str,
    required=False,
)

parser.add_argument(
    "--num",
    default=5,
    type=int,
    required=False,
)
parser.add_argument(
    "--top",
    default=5,
    type=int,
    required=False,
)
parser.add_argument(
    "--model",
    default='bertweet',#'facebook/bart-base',
    type=str,
    required=False,
)

parser.add_argument(
    "--input",
    default='ft_retrisameone20_tmpcon050_iter',
    type=str,
    required=False,
)

with torch.no_grad():
    # cos_sim = torch.nn.CosineSimilarity(dim=1).cuda()
    cos_sim =  torch.nn.PairwiseDistance()
    args = parser.parse_args()
    for task in tqdm(args.task_name.strip().split(',')):
        text = []
        for seed in args.seed.strip().split(','):
            label2idx = CONVERT[task.split('_')[0]]
            num_classes = len(label2idx.keys())
            if not os.path.isdir('./'+args.input+'/'+task+'/'+seed):
                continue
            config = AutoConfig.from_pretrained('./'+args.input+'/'+task+'/'+seed, num_labels=num_classes)

            if 'bart' in args.model:
                model = AutoModelForSequenceClassification.from_pretrained('./' + args.input + '/' + task + '/' + seed, config=config)
                embs = model.model.decoder.embed_tokens
            else:
                model = RobertaForMulti.from_pretrained('./' + args.input + '/' + task + '/' + seed, config=config)
                embs = model.roberta.embeddings.word_embeddings
            # model.model.decoder.embed_tokens
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            vocab = len(tokenizer)
            # words = tokenizer.get_vocab()

            # text_json = {}
            # for idx in range(args.num):
            #     dis = -cos_sim(embs.weight[vocab+idx:vocab+idx+1],embs.weight[:vocab])
            #     val, best_idx = dis.topk(args.top)
            #     text = []
            #     for idx2 in best_idx:
            #         # text += tokenizer._convert_id_to_token(idx2.item()) + ' '
            #         text.append( tokenizer.decode([idx2.item()]) )
            #     text_json[idx] = text
            # with open(args.input + '/' + task +'.json','a',encoding='utf-8') as f:
            #     tmp = json.dumps(text_json, ensure_ascii=False)
            #     f.write(tmp + '\n')

            for idx in range(args.num):
                dis = -cos_sim(embs.weight[vocab + idx:vocab + idx + 1], embs.weight[:vocab])
                val, best_idx = dis.topk(args.top)
                for idx2 in best_idx:
                    # text += tokenizer._convert_id_to_token(idx2.item()) + ' '
                    text.append(tokenizer.decode([idx2.item()]))
        with open(args.input + '/' + task + '.txt', 'w', encoding='utf-8') as f:
            f.write(' '.join(text))
        del embs,model,tokenizer,vocab

