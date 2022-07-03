for SHOT in 16 32 64 128 256 512 1024
do
    CUDA_VISIBLE_DEVICES=5 python prompt_temp.py --results_name "results_lr3e5_epoch15_tempITWASMASK_shot${SHOT}.txt" --template "It was {'mask'}" --num_train_epochs 15 --shot ${SHOT}
done
CUDA_VISIBLE_DEVICES=5 python prompt_temp.py --results_name "results_lr3e5_epoch15_tempITWASMASK.txt" --template "It was {'mask'}" --num_train_epochs 15