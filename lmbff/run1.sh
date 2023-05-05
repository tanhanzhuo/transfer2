for TASK in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor
do
  CUDA_VISIBLE_DEVICES=3 python find_tmp_auto.py --train ../finetune/data/${TASK}/train.tsv --dev ../finetune/data/${TASK}/dev.tsv --template '<s> {sentence} [T] [T] [T] [T] [P] . </s>' --num-cand 100 --accumulation-steps 30 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt4.log --max_seq_length 300 --label-map 3
  CUDA_VISIBLE_DEVICES=3 python find_tmp_auto.py --train ../finetune/data/${TASK}/train.tsv --dev ../finetune/data/${TASK}/dev.tsv --template '<s> {sentence} [T] [T] [T] [P] . </s>' --num-cand 100 --accumulation-steps 30 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt3.log --max_seq_length 300 --label-map 3
  CUDA_VISIBLE_DEVICES=3 python find_tmp_auto.py --train ../finetune/data/${TASK}/train.tsv --dev ../finetune/data/${TASK}/dev.tsv --template '<s> {sentence} [T] [T] [P] . </s>' --num-cand 100 --accumulation-steps 30 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt2.log --max_seq_length 300 --label-map 3
  CUDA_VISIBLE_DEVICES=3 python find_tmp_auto.py --train ../finetune/data/${TASK}/train.tsv --dev ../finetune/data/${TASK}/dev.tsv --template '<s> {sentence} [T] [P] . </s>' --num-cand 100 --accumulation-steps 30 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt1.log --max_seq_length 300 --label-map 3
done