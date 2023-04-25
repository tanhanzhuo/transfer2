import argparse
import logging
# from transformers.utils import logging
import os
import sys
import random
import time
import math
import distutils.util
from functools import partial
import copy
import numpy as np
import datasets
# import paddle
# from paddle.io import DataLoader
import json
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    get_linear_schedule_with_warmup,
    RobertaForMaskedLM
)
from transformers.modeling_outputs import MaskedLMOutput
from accelerate import Accelerator
from tqdm import trange,tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import torch.nn as nn
# import paddle.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaLMHead
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.prompts import SoftVerbalizer
from openprompt.prompts import ManualVerbalizer

MAX_NUM_VECTORS = 20

CONVERT = {
    'eval-emotion':{'0':0,'1':1,'2':2,'3':3},
    'eval-hate':{'0':0,'1':1},
    'eval-irony':{'0':0,'1':1},
    'eval-offensive':{'0':0,'1':1},
    'eval-stance':{'0':0,'1':1,'2':2},
    'stance': {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}


WORDS = {
    # 'stance':[["yes"], ["agree","like","favor"], ["dis","don't","not","hate"]],
    'eval-emotion': [["angerous"], ["joyful"], ["optimistic"],["sad"]],
    'eval-hate': [["neutral"], ["hateful"]],
    'eval-irony': [["neutral"], ["ironic"]],
    'eval-offensive': [["neutral"], ["offensive"]],
    'eval-stance': [["neutral"], ["against"],["favor"]],
    'stance': [["neutral"], ["favor"],["against"]],
    'sem22-task6-sarcasm': [["neutral"], ["sarcastic"]],
    'sem21-task7-humor': [["neutral"], ["humorous"]]
}

TEMPLATE = {
    'eval-emotion':' {"mask"}.',
    'eval-hate':' {"mask"}.',
    'eval-irony':' {"mask"}.',
    'eval-offensive':' {"mask"}.',
    'eval-stance':' {"mask"}.',
    'stance':' {"mask"}.',
    'sem22-task6-sarcasm':' {"mask"}.',
    'sem21-task7-humor':' {"mask"}.'
}


class RobertaForMulti(RobertaForMaskedLM):
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        num_old = self.roberta.config.max_position_embeddings
        if num_old == new_num_position_embeddings:
            return
        self.roberta.config.max_position_embeddings = new_num_position_embeddings
        # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
        new_position = nn.Embedding(self.roberta.config.max_position_embeddings, self.roberta.config.hidden_size)
        new_position.to(self.roberta.embeddings.position_embeddings.weight.device,
                        dtype=self.roberta.embeddings.position_embeddings.weight.dtype)
        # self._init_weights(new_position)
        new_position.weight.data[:num_old, :] = self.roberta.embeddings.position_embeddings.weight.data[:num_old, :]
        self.roberta.embeddings.position_embeddings = new_position

        self.roberta.embeddings.register_buffer("position_ids", torch.arange(self.roberta.config.max_position_embeddings).expand((1, -1)))
        self.roberta.embeddings.register_buffer(
            "token_type_ids", torch.zeros([1,self.roberta.config.max_position_embeddings], dtype=torch.long), persistent=False
        )
        # with torch.no_grad():
        #     # self.roberta.embeddings.position_embeddings.weight[:num_old,:] = nn.Parameter(
        #     #     old_position_embeddings_weight)

    def resize_type_embeddings(self, new_type_embeddings: int):
        num_old = self.roberta.config.type_vocab_size
        if num_old == new_type_embeddings:
            return
        self.roberta.config.type_vocab_size = new_type_embeddings
        # old_position_embeddings_weight = self.roberta.embeddings.position_embeddings.weight.clone()
        new_type = nn.Embedding(self.roberta.config.type_vocab_size, self.roberta.config.hidden_size)
        new_type.to(self.roberta.embeddings.token_type_embeddings.weight.device,
                        dtype=self.roberta.embeddings.token_type_embeddings.weight.dtype)
        # self._init_weights(new_position)
        new_type.weight.data[:num_old, :] = self.roberta.embeddings.token_type_embeddings.weight.data[:num_old, :]
        self.roberta.embeddings.token_type_embeddings = new_type

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--task",
        # default='stance,hate,sem-18,sem-17,imp-hate,sem19-task5-hate,sem19-task6-offen,sem22-task6-sarcasm',
        #default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm,stance',
        default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor',
        type=str,
        required=False,
        help="The name of the task to train selected in the list: ")
    parser.add_argument(
        "--model_name_or_path",
        default='vinai/bertweet-base',
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: "
    )
    parser.add_argument(
        "--token_name_or_path",
        default='vinai/bertweet-base',
        type=str,
        required=False,
    )
    parser.add_argument(
        "--input_dir",
        default='../finetune/data/',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--method",
        default='_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--results_name",
        default='results_all.txt',
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=200,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--token_type",
        default=1,
        type=int)
    parser.add_argument(
        "--learning_rate",
        default='1e-5',#'1e-3,1e-4,1e-5,1e-6',
        type=str,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs",
        default=30,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup",
        help="The scheduler type to use.",
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0.1, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_train_steps",
        default=None,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--seed", default='0,1,2,3,4,5,6,7,8,9', type=str, help="random seed for initialization")
    parser.add_argument(
        "--shot", default='full',#'10,20,40,80,160,320,640,1280,full',
        type=str, help="random seed for initialization")
    parser.add_argument(
        "--stop", default=5, type=int, help="early stop")
    parser.add_argument(
        "--weight", default=0, type=int, help="weighted loss")
    parser.add_argument(
        "--write_result", default='', type=str, help="weighted loss")
    parser.add_argument(
        "--demo", default=1, type=int, help="with demo")
    parser.add_argument(
        "--soft", default=2, type=int, help="soft verberlizer")
    parser.add_argument(
        "--temp", default='5,[X],5,[Y],5', type=str, help="soft template")
    parser.add_argument(
        "--mask", default='text+{"mask"}.', type=str, help="hard template")
    args = parser.parse_args()
    return args

def read_data(fileName):
    data_dic = []
    with open(fileName, 'r') as f:
        for line in f:
            data_dic.append(json.loads(line))
    return data_dic

def assign_embedding(tokenizer, model, new_token, token):
    id_a = tokenizer.convert_tokens_to_ids([new_token])[0]
    id_b = tokenizer.convert_tokens_to_ids([token])[0]
    with torch.no_grad():
        model.roberta.embeddings.word_embeddings.weight[id_a] = \
        model.roberta.embeddings.word_embeddings.weight[id_b].detach().clone()

def get_new_token(vid):
    assert(vid >= 0 and vid < MAX_NUM_VECTORS)
    return '[C_P_%d]'%(vid)

def prepare_for_dense_prompt(model, tokenizer):
    new_tokens = [get_new_token(i) for i in range(MAX_NUM_VECTORS)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

# def convert_manual_to_dense(manual_template, model, tokenizer):
#     new_token_id = 0
#     template = []
#     for word in manual_template.split(','):
#         if word in ['[X]', '[Y]']:
#             template.append(word)
#         else:
#             tokens = tokenizer.tokenize(' ' + word)
#             for token in tokens:
#                 new_token_id += 1
#                 template.append(get_new_token(new_token_id))
#                 assign_embedding(get_new_token(new_token_id), token)
#
#     return ' '.join(template)

def convert_manual_to_dense(manual_template, model, tokenizer):
    new_token_id = 0
    template = []
    for word in manual_template.split(','):
        if word in ['[X]', '[Y]']:
            template.append(word)
        elif word.isdigit():
            tokens = int(word)
            for token in range(tokens):
                template.append(get_new_token(new_token_id))
                new_token_id += 1
        else:
            tokens = tokenizer.tokenize(' ' + word)
            for token in tokens:
                template.append(get_new_token(new_token_id))
                assign_embedding(tokenizer, model, get_new_token(new_token_id), token)
                new_token_id += 1

    return ' '.join(template)


def get_template_text(manual_template, tokenizer):
    new_token_id = 0
    template = []
    for word in manual_template.split(','):
        if word in ['[X]', '[Y]']:
            template.append(word)
        elif word.isdigit():
            tokens = int(word)
            for token in range(tokens):
                template.append(get_new_token(new_token_id))
                new_token_id += 1
        else:
            tokens = tokenizer.tokenize(' ' + word)
            for token in tokens:
                template.append(get_new_token(new_token_id))
                new_token_id += 1
    return ' '.join(template)

@torch.no_grad()
def evaluate(model, data_loader, task='eval-emoji',write_result=''):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        # input_ids, segment_ids, labels = batch
        # logits = model(input_ids.cuda(), segment_ids.cuda())
        logits = model(batch.cuda())
        labels = batch['label']
        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.cpu().numpy()]
        pred_all += [tmp for tmp in preds.cpu().numpy()]
    if len(write_result) > 0:
        with open(write_result, 'a', encoding='utf-8') as f:
            f.write(task+'\n')
            for one in pred_all:
                f.write(str(one))
            f.write('\n')
    results = classification_report(label_all, pred_all, output_dict=True)

    if 'emoji' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Emotion (Macro f1)
    elif 'emotion' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Hate (Macro f1)
    elif 'hate' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Irony (Irony class f1)
    elif 'irony' in task:
        tweeteval_result = results['1']['f1-score']

        # Offensive (Macro f1)
    elif 'offensive' in task:
        tweeteval_result = results['macro avg']['f1-score']

        # Sentiment (Macro Recall)
    elif 'sentiment' in task:
        tweeteval_result = results['macro avg']['recall']

        # Stance (Macro F1 of 'favor' and 'against' classes)
    elif 'stance' in task:
        f1_against = results['1']['f1-score']
        f1_favor = results['2']['f1-score']
        tweeteval_result = (f1_against + f1_favor) / 2
    elif 'sarcasm' in task:
        tweeteval_result = results['1']['f1-score']
    elif 'humor' in task:
        tweeteval_result = results['1']['f1-score']

    print("aveRec:%.5f, f1PN:%.5f, acc: %.5f " % (tweeteval_result, tweeteval_result, tweeteval_result))
    return tweeteval_result,tweeteval_result,tweeteval_result


def do_train(args):
    print(args)
    label2idx = CONVERT[args.task]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True,
                                              model_max_length=args.max_seq_length - 2)
    original_vocab_size = len(list(tokenizer.get_vocab()))
    tokenizer.model_max_length = args.max_seq_length - 2
    tokenizer._pad_token_type_id = args.token_type - 1
    new_tokens = [get_new_token(i) for i in range(MAX_NUM_VECTORS)]
    tokenizer.add_tokens(new_tokens)

    template = get_template_text(args.temp.strip(), tokenizer)
    dataset = {}
    SPLITS = ['train', 'dev', 'test']
    for split in SPLITS:
        dataset[split] = []
        data_all = read_data(args.input_dir + split + args.method + '.json')
        random.shuffle(data_all)
        for data in data_all:
            if args.demo > 0:
                text_demo = ''
                for idx in range(args.demo):
                    text_demo = data['text' + str(idx)] + ' ' + text_demo
                text_final = template.replace('[Y]', data['text']).replace('[X]', text_demo)
                input_example = InputExample(text_a=text_final, label=int(label2idx[data['labels']]))
            else:
                input_example = InputExample(text_a=data['text'], label=int(label2idx[data['labels']]))
            dataset[split].append(input_example)
    ##################few shot
    if args.shot != 'full':
        sampler = FewShotSampler(num_examples_per_label=int(args.shot))
        dataset['train'] = sampler(dataset['train'])
    mask_sp = args.mask.strip().split('+')
    if mask_sp[0] == 'text':
        template_text = '{"placeholder":"text_a"}' + mask_sp[1]  # + TEMPLATE[args.task]
    else:
        template_text = mask_sp[0] + '{"placeholder":"text_a"}'
    mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

    train_data_loader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=MLMTokenizerWrapper,
                                         max_seq_length=args.max_seq_length - 2,
                                         batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                         predict_eos_token=False, truncate_method="head")
    dev_data_loader = PromptDataLoader(dataset=dataset["dev"], template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=MLMTokenizerWrapper,
                                       max_seq_length=args.max_seq_length - 2,
                                       batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="head")
    test_data_loader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=MLMTokenizerWrapper,
                                        max_seq_length=args.max_seq_length - 2,
                                        batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
                                        predict_eos_token=False, truncate_method="head")

    learning_rate = args.learning_rate.split(',')
    ##################### train soft template first
    best_metric = [0, 0, 0]
    for lr in learning_rate:
        best_metric_lr = [0, 0, 0]
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # plm = RobertaForMaskedLM.from_pretrained(
        #     args.model_name_or_path, config=config).cuda()
        plm = RobertaForMulti.from_pretrained(args.model_name_or_path, config=config)
        plm.resize_position_embeddings(args.max_seq_length)
        plm.resize_type_embeddings(args.token_type)
        plm.resize_token_embeddings(len(tokenizer))
        convert_manual_to_dense(args.temp.strip(), plm, tokenizer)
        hard_verb = ManualVerbalizer(tokenizer, num_classes=len(label2idx.keys()),
                                     label_words=WORDS[args.task])
        model = PromptForClassification(plm=plm.cuda(), template=mytemplate, verbalizer=hard_verb, freeze_plm=False)
        model = model.cuda()
        optimizer = AdamW([{'params': model.plm.roberta.embeddings.word_embeddings.parameters()}], lr=float(lr),
                          correct_bias=False)
        num_update_steps_per_epoch = len(train_data_loader)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_steps * args.max_train_steps),
            num_training_steps=args.max_train_steps,
        )
        loss_fct = nn.CrossEntropyLoss().cuda()
        global_step = 0
        tic_train = time.time()

        stop_sign = 0
        for epoch in trange(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                logits = model(batch.cuda())
                loss = loss_fct(logits, batch['label'].cuda().view(-1))
                loss.backward()

                for p in model.plm.roberta.embeddings.word_embeddings.parameters():
                    # only update new tokens
                    p.grad[:original_vocab_size, :] = 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (epoch + 1) % args.logging_steps == 0:
                print(
                    "soft template global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                    % (global_step, args.max_train_steps, epoch,
                       loss, args.logging_steps / (time.time() - tic_train),
                       args.seed, float(lr), args.input_dir))
                tic_train = time.time()
            if (epoch + 1) % args.save_steps == 0 and (epoch + 1) > 3:
                tic_eval = time.time()
                cur_metric = evaluate(model, dev_data_loader, args.task)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric_lr[0]:
                    best_metric_lr = cur_metric
                    stop_sign = 0
                    if best_metric_lr[0] > best_metric[0]:
                        model_best = copy.deepcopy(model).cpu()
                        best_metric = best_metric_lr

                else:
                    stop_sign += 1
            if stop_sign >= args.stop:
                break
        del model
        torch.cuda.empty_cache()

    ##################### normal training
    best_metric = [0, 0, 0]
    for lr in learning_rate:
        best_metric_lr = [0, 0, 0]

        if args.soft == 1:
            myverbalizer = SoftVerbalizer(tokenizer, plm.cpu(), num_classes=len(label2idx.keys()))
        elif args.soft == 2:
            myverbalizer = SoftVerbalizer(tokenizer, plm.cpu(), num_classes=len(label2idx.keys()), label_words=WORDS[args.task])
        else:
            myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(label2idx.keys()),
                                        label_words=WORDS[args.task])
        model = copy.deepcopy(model_best)
        model.verbalizer = myverbalizer
        model = model.cuda()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=float(lr))
        num_update_steps_per_epoch = len(train_data_loader)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_steps*args.max_train_steps),
            num_training_steps=args.max_train_steps,
        )

        loss_fct = nn.CrossEntropyLoss().cuda()
        if args.weight == 1:# or 'sarcasm' in args.task:
            num_dic = {}
            for val in label2idx.values():
                num_dic[val] = 0.0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                num_dic[label_tmp] += 1.0
            num_max = max(num_dic.values())
            class_weights = [num_max / i for i in num_dic.values()]
            class_weights = torch.FloatTensor(class_weights).cuda()
            loss_fct = nn.CrossEntropyLoss(weight=class_weights).cuda()

        print('start Training!!!')
        global_step = 0
        tic_train = time.time()

        stop_sign = 0
        for epoch in trange(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                # input_ids, segment_ids, labels = batch
                # logits = model(input_ids.cuda(), segment_ids.cuda())
                # loss = loss_fct(logits, labels.cuda().view(-1))
                logits = model(batch.cuda())
                loss = loss_fct(logits, batch['label'].cuda().view(-1))
                # print(step)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if (epoch + 1) % args.logging_steps == 0:
                print(
                    "global step %d/%d, epoch: %d, loss: %f, speed: %.4f step/s, seed: %d,lr: %.5f,task: %s"
                    % (global_step, args.max_train_steps, epoch,
                       loss, args.logging_steps / (time.time() - tic_train),
                       args.seed,float(lr),args.input_dir))
                tic_train = time.time()
            if (epoch + 1) % args.save_steps == 0 and (epoch + 1) > 3:
                tic_eval = time.time()
                cur_metric = evaluate(model, dev_data_loader, args.task)
                print("eval done total : %s s" % (time.time() - tic_eval))
                if cur_metric[0] > best_metric_lr[0]:
                    best_metric_lr = cur_metric
                    stop_sign = 0
                    if best_metric_lr[0] > best_metric[0]:
                        model_best_final = copy.deepcopy(model).cpu()
                        best_metric = best_metric_lr

                else:
                    stop_sign += 1
            if stop_sign >= args.stop:
                break
        del model
        torch.cuda.empty_cache()

    model = model_best_final.cuda()
    cur_metric = evaluate(model, test_data_loader,args.task,args.write_result)
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
    del model
    return cur_metric

if __name__ == "__main__":
    args = parse_args()
    for shot in args.shot.split(','):
        for task in args.task.split(','):
            for model_name in args.model_name_or_path.split(','):
                ave_metric = []
                for seed in args.seed.split(','):
                    set_seed(int(seed))
                    args_tmp = copy.deepcopy(args)
                    args_tmp.task = task.split('_')[0]
                    args_tmp.input_dir = args.input_dir + task + '/'
                    args_tmp.seed = int(seed)
                    args_tmp.shot = shot
                    args_tmp.model_name_or_path = model_name
                    ave_metric.append(do_train(args_tmp))
                ave_metric = np.array(ave_metric)
                num_seed = len(args.seed.split(','))
                print("*************************************************************************************")
                print('Task: %s, model: %s, shot: %s' % (task, model_name, shot))
                print('final aveRec:%.5f, f1PN:%.5f, acc: %.5f ' % (sum(ave_metric[:, 0]) / num_seed,
                                                                    sum(ave_metric[:, 1]) / num_seed,
                                                                    sum(ave_metric[:, 2]) / num_seed))
                with open(args.results_name, 'a') as f_res:

                    f_res.write('Task: %s, model: %s, shot: %s\n' % (task, model_name, shot) )
                    f_res.write('aveRec:%.5f, f1PN:%.5f, acc: %.5f \n' % (sum(ave_metric[:, 0]) / num_seed,
                                                                          sum(ave_metric[:, 1]) / num_seed,
                                                                          sum(ave_metric[:, 2]) / num_seed))
                    for tmp in range(num_seed):
                        f_res.write('%.5f, %.5f, %.5f \n' % (ave_metric[tmp, 0],ave_metric[tmp, 1],ave_metric[tmp, 2]))

                    f_res.close()
