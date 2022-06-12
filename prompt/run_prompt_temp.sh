for SEED in 0,1,2
for ORDER in refirst,orifirst
for KTH in 0,1,2
do
do
do
CUDA_VISIBLE_DEVICES=7 python prompt_temp.py --task_name sem-18,sem19-task6-offen --method _emo_hash_process_seed${SEED}_${ORDER}_kth${KTH} --results_name results_emo_hash_seed${SEED}_${ORDER}_kth${KTH}.txt --template "It was {'mask'}" --num_train_epochs 15
done
done
done