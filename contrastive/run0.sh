CUDA_VISIBLE_DEVICES=0 python contrastive_extract_feature_together_cosine.py --file selected_thre100_num500 --model /work/SimCSE-main/result/thre100_num100/299999/ --num_sample 10 --save feature_modelT100N100_fileT100_num100 --CUR_SPLIT 0 --batch_size 7 --preprocessing_num_workers 20
CUDA_VISIBLE_DEVICES=0 python contrastive_extract_feature_together_cosine.py --file selected_thre100_num500 --model /work/SimCSE-main/result/thre100_num100/299999/ --num_sample 10 --save feature_modelT100N100_fileT100_num100 --CUR_SPLIT 1 --batch_size 7 --preprocessing_num_workers 20