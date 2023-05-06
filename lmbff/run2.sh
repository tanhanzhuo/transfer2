#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path vinai/bertweet-base --results_name results_bertweet.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_100/99999/ --results_name results_111_100-99999.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_100/9999/ --results_name results_111_100-9999.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/99999/ --results_name results_111_sep_100-99999.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/9999/ --results_name results_111_sep_100-9999.txt --seed 0,1,2,3,4,5,6,7,8,9

#CUDA_VISIBLE_DEVICES=0 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path vinai/bertweet-base --method hash_fulldata_simcse_top_1_rerank --results_name results_simcse_rerank.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=0 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path vinai/bertweet-base --method hash_fulldata_simcse_top_1 --results_name results_simcse.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9

#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_100/99999/ --method hash_fulldata_simcse_top_1_rerank --results_name results_H111-100-99999_simcse_rerank.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_100/99999/ --method hash_fulldata_simcse_top_1 --results_name results_H111-100-99999_simcse.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/99999/ --method hash_fulldata_simcse_top_1_rerank --results_name results_H111-sep-100-99999_simcse_rerank.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/99999/ --method hash_fulldata_simcse_top_1 --results_name results_H111-sep-100-99999_simcse.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9

#for epoch in {999999..99999..-100000}
#do
#    CUDA_VISIBLE_DEVICES=0 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_100/${epoch}/ --results_name results_111_100_${epoch}.txt --seed 0,1,2,3,4,5,6,7,8,9
#done

#for epoch in 99999 499999 999999
#do
#  CUDA_VISIBLE_DEVICES=1 python prompt_extend_position_tokentype_fs_topic.py --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --model_name_or_path ../pretrain/hashtag/hash_group_111_100/${epoch}/ --results_name results_soft_fs_100_${epoch}.txt --max_seq_length 200 --batch_size 16 --demo 0 --soft 1 --shot 16,32,64,128,256,512,full --seed 0,1,2,3,4,5,6,7,8,9
#  CUDA_VISIBLE_DEVICES=1 python prompt_extend_position_tokentype_fs_topic.py --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --model_name_or_path ../pretrain/hashtag/hash_group_111_20/${epoch}/ --results_name results_soft_fs_20_${epoch}.txt --max_seq_length 200 --batch_size 16 --demo 0 --soft 1 --shot 16,32,64,128,256,512,full --seed 0,1,2,3,4,5,6,7,8,9
#done

#CUDA_VISIBLE_DEVICES=6 python prompt_extend_position_tokentype_fs_topic.py --model_name_or_path ../pretrain/hashtag/hash_group_111_100/999999/ --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --max_seq_length 200 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full --demo 0 --soft 0 --choice 0 --results_name results_gp_hard_old.txt
#CUDA_VISIBLE_DEVICES=6 python prompt_extend_position_tokentype_fs_topic.py --model_name_or_path ../pretrain/hashtag/hash_group_111_100/999999/ --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --max_seq_length 200 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full --demo 0 --soft 0 --choice 1 --results_name results_gp_hard_new.txt
#CUDA_VISIBLE_DEVICES=7 python prompt_extend_position_tokentype_fs_topic.py --model_name_or_path vinai/bertweet-base --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --max_seq_length 200 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full --demo 0 --soft 1 --choice 0 --results_name results_bt_soft_no.txt
#CUDA_VISIBLE_DEVICES=6 python prompt_extend_position_tokentype_fs_topic.py --model_name_or_path ../pretrain/hashtag/hash_group_111_100/999999/ --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --max_seq_length 200 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full --demo 0 --soft 2 --choice 0 --results_name results_gp_soft_old.txt
#CUDA_VISIBLE_DEVICES=6 python prompt_extend_position_tokentype_fs_topic.py --model_name_or_path ../pretrain/hashtag/hash_group_111_100/999999/ --method _modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst_sp --max_seq_length 200 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full --demo 0 --soft 2 --choice 1 --results_name results_gp_soft_new.txt

for TASK in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor
do
  CUDA_VISIBLE_DEVICES=3 python find_tmp_auto.py --train ../finetune/data/${TASK}/train_fuldata_bt_hashseg_top20_textfirst.tsv --dev ../finetune/data/${TASK}/dev_fuldata_bt_hashseg_top20_textfirst.tsv --template '<s> [T] [T] [T] {sentence_B} </s> [T] [T] [T] {sentence_A} . It was [P] . </s>' --num-cand 100 --accumulation-steps 30 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt_retri_ori_33.log --max_seq_length 400
done
for TASK in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor
do
  CUDA_VISIBLE_DEVICES=3 python find_tmp_auto.py --train ../finetune/data/${TASK}/train_fuldata_bt_hashseg_top20_textfirst.tsv --dev ../finetune/data/${TASK}/dev_fuldata_bt_hashseg_top20_textfirst.tsv --template '<s> {sentence_B} </s> [T] [T] [T] {sentence_A} . It was [P] . </s>' --num-cand 100 --accumulation-steps 30 --bsz 24 --eval-size 48 --iters 100 --model-name vinai/bertweet-base --log_name bt_retri_ori_03.log --max_seq_length 400
done
