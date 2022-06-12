for SEED in 0 1 2
do
  for ORDER in refirst orifirst
  do
    for KTH in 0 1 2
    do
      CUDA_VISIBLE_DEVICES=4 python finetune_lrtune_macrof1_fromtxt.py --task_name sem-18,sem19-task6-offen --method "_emo_hash_process_seed${SEED}_${ORDER}_kth${KTH}" --results_name "results_emo_hash_seed${SEED}_${ORDER}_kth${KTH}.txt" --learning_rate 3e-5 --num_train_epochs 15 --batch_size 32
    done
  done
done