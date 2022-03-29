CUDA_VISIBLE_DEVICES=4,5 accelerate launch mlm.py \
--train_file /work/test/pretrain_hashtag/twitter_ref_prob/TrainData_group \
--model_name_or_path vinai/bertweet-base \
--tokenizer_name vinai/bertweet-base \
--output_dir /work/test/pretrain_hashtag/keyphrase/pretrain/tmp_group_1e4/ \
--per_device_train_batch_size 128 \
--gradient_accumulation_steps 8 \
--max_seq_length 128 \
--save_step 5000 \
--num_train_epoch 6 \
--learning_rate 1e-4 \
--weight_decay 1e-2 \
--num_warmup_steps 5000
