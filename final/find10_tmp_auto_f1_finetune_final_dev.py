import time
import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import f1_score
import autoprompt.utils as utils
import time
import math
import distutils.util
from functools import partial
import copy
import numpy as np
import datasets
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
)
from accelerate import Accelerator
from tqdm import trange,tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import torch.nn as nn


logger = logging.getLogger(__name__)

CONVERT = {
    'eval-emotion': {'0': 0, '1': 1, '2': 2, '3': 3},
    'eval-hate': {'0': 0, '1': 1},
    'eval-irony': {'0': 0, '1': 1},
    'eval-offensive': {'0': 0, '1': 1},
    'eval-stance': {'0': 0, '1': 1, '2': 2},
    'stance': {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2},
    'sem22-task6-sarcasm': {'0': 0, '1': 1},
    'sem21-task7-humor': {'0': 0, '1': 1}
}

WORDS = {
    'eval-stance': [["neutral"], ["against"],["favor"]],
    'eval-emotion': [["angerous"], ["joyful"], ["optimistic"],["sad"]],
    'eval-irony': [["neutral"], ["ironic"]],
    'eval-offensive': [["neutral"], ["offensive"]],
    'eval-hate': [["neutral"], ["hateful"]],
    'sem21-task7-humor': [["neutral"], ["humorous"]],
    'sem22-task6-sarcasm': [["neutral"], ["sarcastic"]]
}

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_gradient = None
        # module.register_backward_hook(self.hook)
        module.register_full_backward_hook(self.hook)
    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits = self._model(**model_inputs).logits
        return logits
        # predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        # return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """

    def __init__(self, tokenizer, label_map):
        label_token = []
        for word in label_map.values():
            word_token = tokenizer.convert_tokens_to_ids(word[0])
            label_token.append(word_token)

        self.revert_lab = dict( zip( label_token, [int(i) for i in label_map.keys()] ) )
        self.label_token = torch.tensor(label_token)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        label_convert = []
        for lab in gold_label_ids:
            label_convert.append(self.revert_lab[lab.item()])
        preds = torch.argmax(predict_logits, axis=-1)

        return f1_score(np.array(label_convert),preds.cpu().numpy(),average='macro')

from transformers import RobertaForSequenceClassification
import torch.nn as nn

class RobertaForMulti(RobertaForSequenceClassification):
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

        self.roberta.embeddings.register_buffer("position_ids",
                                                torch.arange(self.roberta.config.max_position_embeddings).expand(
                                                    (1, -1)))
        self.roberta.embeddings.register_buffer(
            "token_type_ids", torch.zeros([1, self.roberta.config.max_position_embeddings], dtype=torch.long),
            persistent=False
        )


def load_pretrained(model_name, max_seq_length=130, num_classes=2):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
    if "bertweet" in model_name:
        model = RobertaForMulti.from_pretrained(model_name, config=config)
        model.resize_position_embeddings(max_seq_length)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, normalization=True)
        tokenizer.model_max_length = max_seq_length - 2
    else:
        raise ValueError("not implemented")
        # model = AutoModelWithLMHead.from_pretrained(model_name)
        # model.eval()
        # tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


# def set_seed(seed: int):
#     """Sets the relevant random seeds."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)
#     torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def get_loss(predict_logits, label_ids, convert):
    for idx in range(len(label_ids)):
        label_ids[idx] = convert[label_ids[idx].item()]
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    # target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
        _isupper = True
    return _isupper


def run_model(args, model=None):
    # set_seed(args.seed)
    args_tmp.train = Path(args_tmp.train)
    args_tmp.dev = Path(args_tmp.dev)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    logger.info('Loading model, tokenizer, etc.')
    if model == None:
        config, model, tokenizer = load_pretrained(args.model_name, args.max_seq_length, len(CONVERT[task].keys()))
        model.to(device)
    else:
        config, model_tmp, tokenizer = load_pretrained(args.model_name, args.max_seq_length, len(CONVERT[task].keys()))
        del model_tmp
        model.eval()
        model.to(device)
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)


    if args.label_map is not None:
        # label_map = json.loads(args.label_map)
        label_map = {}
        with open(task+'.json', 'r', encoding='utf-8') as f:
            label_map_tmp = json.load(f)
        for label_tmp in list(label_map_tmp.keys()):
            label_map[label_tmp] = list(label_map_tmp[label_tmp].keys())[:int(args.label_map)]

        logger.info(f"Label map: {label_map}")
    else:

        label_map = dict( zip( list(CONVERT[task].keys()), WORDS[task] ) )
        logger.info(f"Label map: {label_map}")
    label_token = []
    for word in label_map.values():
        word_token = tokenizer.convert_tokens_to_ids(word[0])
        label_token.append(word_token)

    revert_lab = dict(zip(label_token, [int(i) for i in label_map.keys()]))

    templatizer = utils.TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        use_ctx=args.use_ctx
    )

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
    if label_map:
        evaluation_fn = AccuracyFn(tokenizer, label_map)
    else:
        evaluation_fn = lambda x, y: -get_loss(x, y)

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    if args.perturbed:
        train_dataset = utils.load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        train_dataset = utils.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = utils.load_augmented_trigger_dataset(args.dev, templatizer)
    else:
        dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
    if "bertweet" in args.model_name:
        filter = torch.zeros(tokenizer.vocab_size + 1, dtype=torch.float32, device=device)
    else:
        filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = utils.encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = -1e32
        else:
            for _, label_ids in train_dataset:
                filter[label_ids] = -1e32
        logger.info('Filtering special tokens and capitalized words.')
        for word, idx in tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= tokenizer.vocab_size:
                continue
            # Filter special tokens.
            if idx in tokenizer.all_special_ids:
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32

    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += 1
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = 0
    # Measure elapsed time of trigger search
    start = time.time()
    cur_flip = templatizer.num_trigger_tokens
    for i_e in range(args.iters):

        logger.info(f'Iteration: {i_e}')

        # logger.info('Accumulating Gradient')
        model.zero_grad()

        pbar = tqdm(range(len(train_loader)))
        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:

            # Shuttle inputs to GPU
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            print(model_inputs)
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels, revert_lab).mean()
            loss.backward()

            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / len(train_loader)
            else:
                averaged_grad += grad.sum(dim=0) / len(train_loader)

        # logger.info('Evaluating Candidates')
        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        if templatizer.num_trigger_tokens > 1:
            while token_to_flip == cur_flip:
                token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        cur_flip = token_to_flip
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand,
                                    filter=filter)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        denom = 0

        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            if best_dev_metric == 0:
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, trigger_ids)
                current_score += evaluation_fn(predict_logits, labels)
            denom += 1

            for i, candidate in enumerate(candidates):
                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)
                candidate_scores[i] += eval_metric
        if best_dev_metric == 0:
            best_dev_metric = current_score / (denom + 1e-13)
        best_candidate_score = candidate_scores.max()
        best_candidate_idx = candidate_scores.argmax()
        # temp_trigger = trigger_ids.clone()
        # temp_trigger[:, token_to_flip] = candidates[best_candidate_idx]
        trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
        dev_metric = best_candidate_score / (denom + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            # trigger_ids = temp_trigger.clone()
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    final_txt = task + '*' + args.template[4:-5]
    final_txt.replace('[P]', '{"mask"}')
    final_txt.replace('{sentence_B}', '{"placeholder":"text_b"}')
    final_txt.replace('{sentence_A}', '{"placeholder":"text_a"}')
    final_txt.replace('{sentence}', '{"placeholder":"text_a"}')
    for token in best_trigger_tokens:
        final_txt = final_txt.replace('[T]', token, 1)
    logger.info(final_txt)
    with open(args.log_name.split('.')[0]+'.txt', 'a', encoding='utf-8') as f:
        f.write(final_txt+'\n')

    return best_trigger_ids.squeeze(0).tolist()



@torch.no_grad()
def evaluate(model, data_loader, task='eval-emoji',write_result=''):
    model.eval()
    label_all = []
    pred_all = []
    for batch in data_loader:
        # input_ids, segment_ids, labels = batch
        # logits = model(input_ids.cuda(), segment_ids.cuda())
        logits = model(input_ids=batch['input_ids'].cuda(),
                       token_type_ids=batch['token_type_ids'].cuda(),
                       attention_mask=batch['attention_mask'].cuda()).logits
        labels = batch['labels']
        preds = logits.argmax(axis=1)
        label_all += [tmp for tmp in labels.numpy()]
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

def convert_example(example, label2idx):
    if example.get('special_tokens_mask') is not None:
        example.pop('special_tokens_mask')
    example['labels'] = label2idx[example['labels']]
    return example  # ['input_ids'], example['token_type_ids'], label, prob

from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    template: Optional[list] = None

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for idx_fea in range(len(features)):
            feature = features[idx_fea]
            flat_features.append({k: feature[k][0] if k in special_keys else feature[k] for k in feature})
            if self.template is None:
                flat_features[idx_fea]['input_ids'] = sum(feature['input_ids'][1:],flat_features[idx_fea]['input_ids'])
            else:
                flat_features[idx_fea]['input_ids'] = sum(feature['input_ids'][1:], flat_features[idx_fea]['input_ids'] + self.template)
            flat_features[idx_fea]['attention_mask'] = flat_features[idx_fea]['attention_mask'] + \
                                                       [1]*(len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['attention_mask']))
            flat_features[idx_fea]['token_type_ids'] = flat_features[idx_fea]['token_type_ids'] + \
                                                       [self.tokenizer.pad_token_type_id] * (len(flat_features[idx_fea]['input_ids']) - len(flat_features[idx_fea]['token_type_ids']))

            # for i in range(num_sent):
                # flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
        #          for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]

        return batch


def do_train(args, model=None):
    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=10)
    # # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # model = RobertaForMulti.from_pretrained(
    #     args.model_name_or_path, config=config).cuda()
    # set_seed(args.seed)
    print(args)
    data_all = datasets.load_from_disk(args.input_dir)
    label2idx = CONVERT[args.task.split('_')[0]]
    trans_func = partial(
        convert_example,
        label2idx=label2idx)
    train_ds = data_all['train']
    train_ds = train_ds.map(trans_func)
    if len(args.shot) > 0:
        if args.shot != 'full':
            sample_num = int(args.shot)
            train_ds = train_ds.shuffle()
            select_idx = []
            select_idx_dic = {}
            for val in label2idx.values():
                select_idx_dic[val] = 0
            for idx in range(len(train_ds)):
                label_tmp = train_ds[idx]['labels']
                if select_idx_dic[label_tmp] < sample_num:
                    select_idx.append(idx)
                    select_idx_dic[label_tmp] += 1
            np.random.shuffle(select_idx)
            train_ds = train_ds.select(select_idx)

    dev_ds = data_all['dev']
    dev_ds = dev_ds.map(trans_func)
    test_ds = data_all['test']
    test_ds = test_ds.map(trans_func)

    learning_rate = args.learning_rate.split(',')
    best_metric = [0, 0, 0]
    model_best = None
    for lr in learning_rate:
        best_metric_lr = [0, 0, 0]
        num_classes = len(label2idx.keys())
        if 'bertweet' in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, normalization=True)
        tokenizer._pad_token_type_id = args.token_type - 1
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_classes)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if model == None:
            model = RobertaForMulti.from_pretrained(
                args.model_name_or_path, config=config).cuda()
            model.resize_position_embeddings(args.max_seq_length)
            # model.resize_type_embeddings(args.token_type)
            if args.finetune_mask == 1:
                batchify_fn = OurDataCollatorWithPadding(tokenizer=tokenizer, \
                                                         template=[tokenizer.mask_token_id]*args.template.count('[T]'))
            else:
                batchify_fn = OurDataCollatorWithPadding(tokenizer=tokenizer)
        else:
            model = model.cuda()
            batchify_fn = OurDataCollatorWithPadding(tokenizer=tokenizer, template=args.template)
            if args.tmp_noeval == 1:
                test_data_loader = DataLoader(
                    test_ds, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
                )
                cur_metric = evaluate(model, test_data_loader, args.task, args.write_result)
                del model
                print('no eval, final')
                print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))
                return cur_metric, model_best

        train_data_loader = DataLoader(
            train_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        dev_data_loader = DataLoader(
            dev_ds, shuffle=True, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        test_data_loader = DataLoader(
            test_ds, shuffle=False, collate_fn=batchify_fn, batch_size=args.batch_size
        )
        print('data ready!!!')
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
                logits = model(input_ids=batch['input_ids'].cuda(),
                               token_type_ids = batch['token_type_ids'].cuda(),
                               attention_mask=batch['attention_mask'].cuda() ).logits
                loss = loss_fct(logits, batch['labels'].cuda().view(-1))
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
            if (epoch + 1) % args.save_steps == 0 and (epoch + 1) > args.eval_start:
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
    if model_best is None:
        cur_metric = [0.0,0.0,0.0]
    else:
        model = model_best.cuda()
        cur_metric = evaluate(model, test_data_loader,args.task,args.write_result)
        del model
    print('final')
    print("f1macro:%.5f, acc:%.5f, acc: %.5f, " % (best_metric[0], best_metric[1], best_metric[2]))
    print("f1macro:%.5f, acc:%.5f, acc: %.5f " % (cur_metric[0], cur_metric[1], cur_metric[2]))

    return cur_metric, model_best

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', type=Path, required=True, help='Train data path')
    # parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str,
                        default='<s> {sentence_A} </s> [T] [T] [T] <s> {sentence_B} </s>',
                        help='Template string')
    parser.add_argument('--label-map', type=str,default=None,#'{"0": ["Ġworse", "Ġincompetence", "ĠWorse", "Ġblamed", "Ġsucked"], "1": ["ĠCris", "Ġmarvelous", "Ġphilanthrop", "Ġvisionary", "Ġwonderful"]}',
                        help='JSON object defining label map')

    # LAMA-specific
    parser.add_argument('--tokenize-labels', action='store_true',
                        help='If specified labels are split into word pieces.'
                             'Needed for LAMA probe experiments.')
    parser.add_argument('--filter', action='store_false',
                        help='If specified, filter out special tokens and gold objects.'
                             'Furthermore, tokens starting with capital '
                             'letters will not appear in triggers. Lazy '
                             'approach for removing proper nouns.')
    parser.add_argument('--print-lama', action='store_true',
                        help='Prints best trigger in LAMA format.')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=16, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=48, help='Eval size')
    parser.add_argument('--iters', type=int, default=30,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bertweet',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences for relation extraction only')
    parser.add_argument('--perturbed', action='store_true',
                        help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=20)
    parser.add_argument('--sentence-size', type=int, default=50)
    parser.add_argument('--log_name', type=str, default='ft_retrione20_03.log')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--max_seq_length', type=int, default=300)
    parser.add_argument('--seed', type=str, default='0,1,2,3,4,5,6,7,8,9')


    parser.add_argument(
        "--task_name",
        default='eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm',
        type=str,required=False)
    parser.add_argument(
        "--model_name_or_path",
        default='bertweet',
        type=str,required=False)
    parser.add_argument(
        "--token_name_or_path",
        default='bertweet',
        type=str,required=False)
    parser.add_argument(
        "--input_dir",
        default='../finetune/data/',
        type=str,required=False)
    parser.add_argument(
        "--method",
        default='hash_seg_500_one20_top_1',
        type=str,required=False)
    parser.add_argument(
        "--results_name",
        default='results_ft_retrione20.txt',
        type=str,required=False)
    # parser.add_argument(
    #     "--max_seq_length",
    #     default=200,
    #     type=int)
    parser.add_argument(
        "--token_type",
        default=1,type=int)
    parser.add_argument(
        "--learning_rate",
        default='1e-5',#'1e-3,1e-4,1e-5,1e-6',
        type=str)
    parser.add_argument(
        "--num_train_epochs",
        default=30,
        type=int)
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1)
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int)
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant_with_warmup"
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps",
        default=0.1,
        type=float)
    parser.add_argument(
        "--max_train_steps",
        default=None,
        type=int)
    # parser.add_argument(
    #     "--seed",
    #     default='0,1,2,3,4,5,6,7,8,9',
    #     type=str)
    parser.add_argument(
        "--shot",
        default='full',
        type=str)
    parser.add_argument(
        "--stop",
        default=5,
        type=int)
    parser.add_argument(
        "--eval_start",
        default=3,
        type=int)
    parser.add_argument(
        "--weight",
        default=0,
        type=int)
    parser.add_argument(
        "--tmp_noeval",
        default=0,
        type=int)
    parser.add_argument(
        "--finetune_mask",
        default=0,
        type=int)
    parser.add_argument(
        "--rerank",
        default='',
        type=str)
    parser.add_argument(
        "--write_result",
        default='',
        type=str)
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    fHandler = logging.FileHandler(args.log_name, mode='a')
    fHandler.setLevel(logging.INFO)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    # logging.basicConfig(filename=args.log_name, level=level)
    logger.info(args)

    for shot in args.shot.split(','):
        for task in args.task_name.split(','):
            for model_name in args.model_name_or_path.split(','):  # [r_dir+'bertweet/']:
                ave_metric = []
                for seed in args.seed.split(','):
                    #train first
                    set_seed(int(seed))
                    args_tmp = copy.deepcopy(args)
                    args_tmp.task = task
                    args_tmp.input_dir = args.input_dir + task + '/' + args.method +args.rerank
                    args_tmp.seed = int(seed)
                    args_tmp.shot = shot
                    args_tmp.model_name_or_path = model_name
                    ave_metri_one, model = do_train(args_tmp)
                    if model == None:
                        ave_metric.append([0.0,0.0,0.0])
                        continue

                    #get template
                    # 'hash_seg_500_one20_top_1'
                    # '../finetune/data/${TASK}/train_seg_500_three20_top100_sp.tsv'

                    args_tmp.train =args.input_dir + task + '/' + \
                                    'train' + args.method.split('hash')[1].split('top')[0] + 'top100_sp'+args.rerank+'.tsv'
                    # args_tmp.train = Path(args_tmp.train)
                    args_tmp.dev = args.input_dir + task + '/' + \
                                     'dev' + args.method.split('hash')[1].split('top')[0] + 'top100_sp'+args.rerank+'.tsv'
                    # args_tmp.dev = Path(args_tmp.dev)
                    tmp = run_model(args_tmp, model)

                    ##train again
                    args_tmp.template = tmp
                    args_tmp.eval_start = 0
                    ave_metri_one, model = do_train(args_tmp,model=model)

                    ave_metric.append(ave_metri_one)
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




