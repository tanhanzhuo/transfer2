import paddle
import torch
import numpy as np
import os
import shutil
# CUDA_VISIBLE_DEVICES=2 python convert_model_final.py
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# paddle.set_device('gpu:4')

for ITER in range(60000, 110000, 5000):
    torch_model_path = "/work/test/pretrain_hashtag/keyphrase/pretrain/tmp_group_3e5/" + str(ITER) + "/pytorch_model.bin"
    torch_state_dict = torch.load(torch_model_path)
    save_path = "/work/test/finetune_newdata/convert_model/pd_group_3e5/" + str(ITER) +'_3e5'
    paddle_model_path = save_path + "/model_state.pdparams"
    paddle_state_dict = {}

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        shutil.copyfile('/work/test/pretrain_hashtag/keyphrase/bertweet/model_config.json', 
                        save_path + "/model_config.json")

    # State_dict's keys mapping: from torch to paddle
    keys_dict = {
        "bert.":"",
        # about embeddings
        "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",

        # about encoder layer
        'encoder.layer': 'encoder.layers',
        'attention.self.query': 'self_attn.q_proj',
        'attention.self.key': 'self_attn.k_proj',
        'attention.self.value': 'self_attn.v_proj',
        'attention.output.dense': 'self_attn.out_proj',
        'attention.output.LayerNorm': 'norm1',
        'intermediate.dense': 'linear1',
        'output.dense': 'linear2',
        'output.LayerNorm': 'norm2',

        # about cls predictions
        'cls.predictions.transform.dense': 'cls.predictions.transform',
        'cls.predictions.decoder.weight': 'cls.predictions.decoder_weight',
        'cls.predictions.transform.LayerNorm': 'cls.predictions.layer_norm',
        'cls.predictions.bias': 'cls.predictions.decoder_bias'
    }


    for torch_key in torch_state_dict:
        if 'position_ids' in torch_key or 'cls' in torch_key:
            continue
        paddle_key = torch_key
        for k in keys_dict:
            if k in paddle_key:
                paddle_key = paddle_key.replace(k, keys_dict[k])

        if ('linear' in paddle_key) or ('proj' in  paddle_key) or ('vocab' in  paddle_key and 'weight' in  paddle_key) or ("dense.weight" in paddle_key) or ('transform.weight' in paddle_key) or ('seq_relationship.weight' in paddle_key):
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy().transpose())
        else:
            paddle_state_dict[paddle_key] = paddle.to_tensor(torch_state_dict[torch_key].cpu().numpy())

        # print("torch: ", torch_key,"\t", torch_state_dict[torch_key].shape)
        # print("paddle: ", paddle_key, "\t", paddle_state_dict[paddle_key].shape, "\n")

    paddle.save(paddle_state_dict, paddle_model_path)

