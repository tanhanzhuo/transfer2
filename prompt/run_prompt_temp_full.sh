CUDA_VISIBLE_DEVICES=2 python prompt_temp.py --method _modelT100N100_fileT100_0 --results_name "results_lr3e5_epoch15_tempITWASMASK_modelT100N100_fileT100_K1.txt" --template "It was {'mask'}" --num_train_epochs 15
CUDA_VISIBLE_DEVICES=2 python prompt_temp.py --method _modelT100N100_fileT100_1 --results_name "results_lr3e5_epoch15_tempITWASMASK_modelT100N100_fileT100_K2.txt" --template "It was {'mask'}" --num_train_epochs 15