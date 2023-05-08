GPU_NUMBER=0
GPU_MEMORY=80

cd ..


for ft in ecthr_a ecthr_b eurlex scotus ledgar unfair_tos case_hold
	do
	echo $ft
	python main.py -gn {GPU_NUMBER} -gm {GPU_MEMORY} -t $ft -los 1,2,3,4,5 --lower_case true -bz 8 -mfbm micro-f1 -nte 20 -lr 3e-5 -esp 3 -ld paper_results
	done
