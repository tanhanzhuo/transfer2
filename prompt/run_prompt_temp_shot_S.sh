for SHOT in 16 32 64 128 256 512 1024
do
  for KTH in 0
  do
    CUDA_VISIBLE_DEVICES=2 python prompt_temp.py --method "_modelT100N100S_fileT100SS_${KTH}" --results_name "results_modelT100N100S_fileT100SS_${KTH}_shot${SHOT}.txt" --template "It was {'mask'}" --num_train_epochs 15 --shot ${SHOT}
  done
done