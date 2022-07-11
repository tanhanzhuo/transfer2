for SHOT in 16 32 64 128 256 512 1024
do
  for KTH in 1
  do
    CUDA_VISIBLE_DEVICES=3 python prompt_temp.py --method "modelT100N100S_fileT100S_top${KTH}_hashlast" --results_name "modelT100N100S_fileT100S_top${KTH}_hashlast_shot${SHOT}.txt" --template "It was {'mask'}" --num_train_epochs 15 --shot ${SHOT}
  done
done