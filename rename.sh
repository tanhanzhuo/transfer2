for TASK in stance hate sem-18 sem-17 imp-hate sem19-task5-hate sem19-task6-offen sem22-task6-sarcasm
do
	for SP in train dev test
	do
		for HW in hash word
		do
			for ORDER in first last
			do
				mv "./finetune/data/${TASK}/${SP}_modelT100N100S_fileT100S_top0 ${HW}${ORDER}.json" "./finetune/data/${TASK}/${SP}_modelT100N100S_fileT100S_top0_${HW}${ORDER}.json"
				mv "./finetune/data/${TASK}/${SP}_modelT100N100S_fileT100S_top1 ${HW}${ORDER}.json" "./finetune/data/${TASK}/${SP}_modelT100N100S_fileT100S_top1_${HW}${ORDER}.json"
			done
		done
	done
done