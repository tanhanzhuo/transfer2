#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path vinai/bertweet-base --results_name results_bertweet.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_100/99999/ --results_name results_111_100-99999.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_100/9999/ --results_name results_111_100-9999.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/99999/ --results_name results_111_sep_100-99999.txt --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=3 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/9999/ --results_name results_111_sep_100-9999.txt --seed 0,1,2,3,4,5,6,7,8,9

#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path vinai/bertweet-base --method hash_fulldata_simcse_top_1_rerank --results_name results_simcse_rerank.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path vinai/bertweet-base --method hash_fulldata_simcse_top_1 --results_name results_simcse.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9

#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_100/99999/ --method hash_fulldata_simcse_top_1_rerank --results_name results_H111-100-99999_simcse_rerank.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_100/99999/ --method hash_fulldata_simcse_top_1 --results_name results_H111-100-99999_simcse.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/99999/ --method hash_fulldata_simcse_top_1_rerank --results_name results_H111-sep-100-99999_simcse_rerank.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9
#CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --model_name_or_path ../pretrain/hashtag/hash_group_111_sep_100/99999/ --method hash_fulldata_simcse_top_1 --results_name results_H111-sep-100-99999_simcse.txt --token_type 1 --learning_rate 1e-5 --shot full --weight 0 --seed 0,1,2,3,4,5,6,7,8,9

#for epoch in {999999..99999..-100000}
#do
#    CUDA_VISIBLE_DEVICES=1 python finetune_lrtune_fs_early.py --task_name eval-stance_clean,eval-emotion_clean,eval-irony_clean,eval-offensive_clean,eval-hate_clean,sem21-task7-humor_clean,sem22-task6-sarcasm_clean --learning_rate 1e-5 --shot full --model_name_or_path ../pretrain/hashtag/hash_group_111_20/${epoch}/ --results_name results_111_20_${epoch}.txt --seed 0,1,2,3,4,5,6,7,8,9
#done

#for epoch in 99999 199999
#do
#  CUDA_VISIBLE_DEVICES=5 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor --model_name_or_path ../pretrain/hashtag/hash_group_111_rep0_100/${epoch}/ --method hash_fuldata_bt_hashseg_top_1 --results_name results_ft_fs_${epoch}_retri.txt --max_seq_length 514 --token_type 1 --learning_rate 1e-5 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full
#  CUDA_VISIBLE_DEVICES=5 python finetune_lrtune_fs_early.py --task_name eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor --model_name_or_path ../pretrain/hashtag/hash_group_111_rep0_100/${epoch}/ --results_name results_ft_fs_${epoch}.txt --learning_rate 1e-5 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot 16,32,64,128,256,512,full
#done

for epoch in 10 18
do
  CUDA_VISIBLE_DEVICES=4 python finetune_lrtune_fs_early.py --task_name eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor --model_name_or_path ../lmbff/contrastive_models/three/${epoch}_new/ --results_name results_ft_fs_contrastivethree${epoch}.txt --learning_rate 1e-5 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot full
  #CUDA_VISIBLE_DEVICES=4 python finetune_lrtune_extend_position_tokentype_fs.py --task_name eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor --model_name_or_path .../lmbff/contrastive_models/three/${epoch}_new/ --method hash_fuldata_bt_hashseg_top_1 --results_name results_ft_fs_contrastivethree${epoch}_retri.txt --max_seq_length 514 --token_type 1 --learning_rate 1e-5 --batch_size 16 --seed 0,1,2,3,4,5,6,7,8,9 --shot full
done
