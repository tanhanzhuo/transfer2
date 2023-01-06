for task in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor sem22-task6-sarcasm stance
do
#  rm -rf /home/thz/transfer2/finetune/data/${task}/hash_10*
  scp thz@10.21.4.61:/home/thz/transfer2/finetune/data/${task}/\{train_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst.json,train_fulldata_simcse_top20_textfirst.json,dev_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst.json,dev_fulldata_simcse_top20_textfirst.json,test_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst.json,test_fulldata_simcse_top20_textfirst.json\} /home/thz/transfer2/finetune/data/${task}/

  #for method in modelT100N100R_fileT100N100R_num10 fulldata_simcse fulldata_bt_hashremove
  #do
    #scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/${task}/hash_${method}_top_* /home/thz/transfer2/finetune/data/${task}/
  #done
done

#scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/sem22-task6-sarcasm/hash_modelT100N100R_fileT100N100R_num10_top_* transfer2/finetune/data/sem22-task6-sarcasm/