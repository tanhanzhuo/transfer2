for task in stance hate sem-18 sem-17 imp-hate sem19-task5-hate sem19-task6-offen sem22-task6-sarcasm sem18-task1-affect sem21-task7-humor
do
  for sp in train dev test
    do
    scp -r thz@10.21.4.51:/home/thz/transfer2/finetune/data/${task}/${sp}_modelT100N100M_fileT100N100S_num10_cluster_top20_textfirst.json ./finetune/data/${task}/
    done
done