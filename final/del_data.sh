for TASK in eval-stance eval-emotion eval-irony eval-offensive eval-hate sem21-task7-humor sem22-task6-sarcasm
do
  rm -rf ../finetune/data/${TASK}_evensplit
  rm -rf ../finetune/data/${TASK}_clean_*
done