for TASK in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor sem22-task6-sarcasm
do
  CUDA_VISIBLE_DEVICES=1 python find1_tmp_auto.py --train ../finetune/data/${TASK}/train_seg_500_one20_top100_sp_rerank.tsv --dev ../finetune/data/${TASK}/dev_seg_500_one20_top100_sp_rerank.tsv --template '<s> {sentence_B} </s> [T] [T] [T] {sentence_A} . It was [P] . </s>' --num-cand 100 --accumulation-steps 50 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt_retrirerank_ori_03.log --max_seq_length 400 --filter
done