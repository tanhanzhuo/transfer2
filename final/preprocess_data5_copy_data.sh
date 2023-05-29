#for task in stance sem-18 sem19-task5-hate sem19-task6-offen sem22-task6-sarcasm sem18-task1-affect sem21-task7-humor
for task in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor sem22-task6-sarcasm
do
  for sp in train dev test
    do
#    scp -r thz@10.21.4.51:/home/thz/transfer2/finetune/data/${task}/${sp}_seg_500_one20_top100_sp* ../finetune/data/${task}/
#      scp -r thz@10.21.4.51:/home/thz/transfer2/finetune/data/${task}/${sp}_seg_500_three20_top100_sp* ../finetune/data/${task}/
#    scp -r thz@10.21.4.51:/home/thz/transfer2/finetune/data/${task}/${sp}_seg_500_simcse_top100_sp* ../finetune/data/${task}/
    scp -r thz@10.21.4.51:/home/thz/transfer2/finetune/data/${task}/${sp}_same_500_simcse_top100_sp.json ../finetune/data/${task}/
    done
done