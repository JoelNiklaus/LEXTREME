#!/bin/bash
#SBATCH --job-name="LEXTREME"
#SBATCH --mail-user=ronja.stern@students.unibe.ch
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --qos=job_gpu_preempt
#SBATCH --partition=gpu
#SBATCH --array=2-4

# enable this for multiple GPUs for max 24h
###SBATCH --time=24:00:00
###SBATCH --qos=job_gpu_preempt
###SBATCH --partition=gpu

# enable this to get a time limit of 20 days
###SBATCH --time=20-00:00:00
###SBATCH --qos=job_gpu_stuermer
###SBATCH --partition=gpu-invest

# load modules
module load Workspace Anaconda3/2021.11-foss-2021a CUDA/11.3.0-GCC-10.2.0

# Activate correct conda environment
eval "$(conda shell.bash hook)"
conda activate scp-test

# Put your code below this line

# distilbert-base-multilingual-cased
# xlm-roberta-base
# for testing use following:
# python main.py -as -bz -gn -gm 24 -hier True -lol -lc -lmt xlm-roberta-base -los 1 -lr -nte -rmo experimental -dmo -t criticality_prediction -ld results/test_scp -nw -cad
python main.py -gm 24 -hier True -lmt joelito/legal-xlm-roberta-large -los 1 -rmo experimental -t tuning
# python main.py -gm 24 -hier True -lmt joelito/legal-xlm-roberta-large -los 1 -rmo experimental -t criticality_prediction
# python main.py -gm 24 -hier True -lmt joelito/legal-xlm-roberta-large -los 1 -rmo experimental -t citation_prediction
# possible args -as -bz -es -gn -gm -hier -ls -lol -lc -lmt -los -nte -rev -rmo -dmo -t -ld -nw -cad -ss -est -lst -sst

# IMPORTANT:
# Run with                  sbatch run_hpc_job.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=gpu-invest --gres=gpu:rtx3090:1 --mem=64G --cpus-per-task=8 --time=02:00:00 --pty /bin/bash
