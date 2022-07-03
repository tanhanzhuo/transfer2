for SHOT in 16 32 64 128 256 512 1024
do
  for KTH in 0 1
  do
    CUDA_VISIBLE_DEVICES=0 python prompt_temp.py --method "_modelT100N100_fileT100_${KTH}" --results_name "results_modelT100N100_fileT100_${KTH}_shot${SHOT}.txt" --template "It was {'mask'}" --num_train_epochs 15 --shot ${SHOT}
  done
done