for task in stance hate sem-18 sem-17 imp-hate sem19-task5-hate sem19-task6-offen sem22-task6-sarcasm
do
  for method in modelT100N100R_fileT100N100R_num10 modelT100N100R_fileT100N1000R_num10 fulldata_simcse fulldata_bt_hashremove
  do
    scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/${task}/hash_${method}_top_* transfer2/finetune/data/${task}/
  done
done

#scp -r thz@10.21.4.61:/home/thz/transfer2/finetune/data/sem22-task6-sarcasm/hash_modelT100N100R_fileT100N100R_num10_top_* transfer2/finetune/data/sem22-task6-sarcasm/