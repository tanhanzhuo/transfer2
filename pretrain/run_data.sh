python create_data.py \
--train_file /work/test/sep/2013_sep0.txt \
--model_name_or_path nghuyong/ernie-2.0-en \
--tokenizer_name nghuyong/ernie-2.0-en \
--output_dir /work/test/hf/collator/hf/sep/ \
--max_seq_length 128 \
--preprocessing_num_workers 10
--use_slow_tokenizer