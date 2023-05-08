GPU_NUMBER=X
GPU_MEMORY=X

cd ..

for finetuning_task in [ecthr_a,ecthr_b, eurlex,scotus,ledgar,unfair_tos, case_hold]
	do
		python main.py -gn {GPU_NUMBER} -gm {GPU_MEMORY} -t finetuning_task -los 1,2,3,4,5 --do_lower_case -bz 8 -mfbm micro-f1 -nte 20 -lr 3e-5 -esp 3
	done
