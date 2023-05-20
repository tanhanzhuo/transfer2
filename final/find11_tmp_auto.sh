for TASK in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor sem22-task6-sarcasm
do
  CUDA_VISIBLE_DEVICES=2 python find10_tmp_auto_f1.py --train ../finetune/data/${TASK}/train_seg_500_three20_top100_sp.tsv --dev ../finetune/data/${TASK}/dev_seg_500_three20_top100_sp.tsv --template '<s> {sentence_B} </s> [T] [T] [T] {sentence_A} . It was [P] . </s>' --num-cand 10 --accumulation-steps 50 --bsz 16 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt_retrithree_label_03.log --max_seq_length 400 --filter
done