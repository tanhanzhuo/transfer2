for task in stance hate sem-18 sem-17 imp-hate sem19-task5-hate sem19-task6-offen sem22-task6-sarcasm sem18-task1-affect sem21-task7-humor
do
  rm -rf /home/thz/transfer2/finetune/data/${task}/emojiorhash_*
  rm -rf /home/thz/transfer2/finetune/data/${task}/emojihash_*
  rm -rf /home/thz/transfer2/finetune/data/${task}/emoji_*
  rm -rf /home/thz/transfer2/finetune/data/${task}/model*
  rm -rf /home/thz/transfer2/finetune/data/${task}/feature_*
  rm -rf /home/thz/transfer2/finetune/data/${task}/hash_10*
  #scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/${task}/hash_* /home/thz/transfer2/finetune/data/${task}/
  #for method in modelT100N100R_fileT100N100R_num10 fulldata_simcse fulldata_bt_hashremove
  #do
    #scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/${task}/hash_${method}_top_* /home/thz/transfer2/finetune/data/${task}/
  #done
done

#scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/sem22-task6-sarcasm/hash_modelT100N100R_fileT100N100R_num10_top_* transfer2/finetune/data/sem22-task6-sarcasm/