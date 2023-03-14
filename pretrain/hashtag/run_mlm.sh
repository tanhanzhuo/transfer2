accelerate launch mlm.py \
--train_file /work/transfer2/pretrain/hashtag/tweet_hash_clean_group_111_100_shuf.txt \
--validation_split_percentage 1 \
--model_name_or_path vinai/bertweet-base \
--tokenizer_name vinai/bertweet-base \
--output_dir /work/transfer2/pretrain/hashtag/hash_group_111_100/ \
--max_seq_length 128 \
--preprocessing_num_workers 20 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--gradient_accumulation_steps 8 \
--num_train_epoch 10 \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--num_warmup_steps 5000 \
--line_by_line True \
--with_tracking \
--report_to wandb \
--save_step 10000


#CUDA_VISIBLE_DEVICES=0 accelerate launch mlm.py --train_file /work/transfer2/pretrain/hashtag/tweet_hash_clean_group_111_100_shuf.txt --validation_split_percentage 1 --model_name_or_path vinai/bertweet-base --tokenizer_name vinai/bertweet-base --output_dir /work/transfer2/pretrain/hashtag/hash_group_111_100/ --max_seq_length 128 --preprocessing_num_workers 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 32 --gradient_accumulation_steps 8 --num_train_epoch 10 --learning_rate 1e-5 --weight_decay 0.0 --num_warmup_steps 5000 --line_by_line True --with_tracking --report_to wandb --save_step 10000
#CUDA_VISIBLE_DEVICES=1 accelerate launch mlm.py --train_file /work/transfer2/pretrain/hashtag/tweet_hash_clean_group_111_sep_100_shuf.txt --validation_split_percentage 1 --model_name_or_path vinai/bertweet-base --tokenizer_name vinai/bertweet-base --output_dir /work/transfer2/pretrain/hashtag/hash_group_111_sep_100/ --max_seq_length 128 --preprocessing_num_workers 20 --per_device_train_batch_size 64 --per_device_eval_batch_size 32 --gradient_accumulation_steps 8 --num_train_epoch 10 --learning_rate 1e-5 --weight_decay 0.0 --num_warmup_steps 5000 --line_by_line True --with_tracking --report_to wandb --save_step 10000