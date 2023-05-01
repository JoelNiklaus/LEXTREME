#!/bin/bash

dataset_name="$1"
model="$2"
seeds="$3"

# Determine the GPU memory based on the dataset name and model
gpu_memory="80"
if [[ $dataset_name == *"facts"* ]]; then
  if [[ $model == *"-base" ]]; then
    gpu_memory="24"
  elif [[ $model == *"-large" ]]; then
    gpu_memory="48"
  fi
elif [[ $dataset_name == *"considerations"* ]]; then
  if [[ $model == *"-base" ]]; then
    gpu_memory="32"
  elif [[ $model == *"-large" ]]; then
    gpu_memory="80"
  fi
fi

# add exceptions
if [[ $dataset_name == "swiss_judgment_prediction_xl_considerations" ]]; then
  gpu_memory="80" # we get a time limit error otherwise
fi

# Replace slashes in the model name with underscores
model_for_path=$(echo "${model}" | tr '/' '_')

# Generate a unique job file name
job_file="jobfiles/${dataset_name}_${model_for_path}_${gpu_memory}.sbatch"

# Create the job file
cat > "${job_file}" << EOL
#!/bin/bash
#SBATCH --job-name=${dataset_name}_${model_for_path}_${gpu_memory}
#SBATCH --output=slurm_logs/${dataset_name}_${model_for_path}_${gpu_memory}.out
#SBATCH --error=slurm_logs/${dataset_name}_${model_for_path}_${gpu_memory}.err
#SBATCH --mail-user=jniklaus@stanford.edu
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH -C GPU_MEM:${gpu_memory}GB
#SBATCH --gpus=1
#SBATCH --partition=owners
#SBATCH --array=${seeds}

# Load necessary modules
ml load python/3.9.0 cuda/11.3.1

# set up the environment
cd /home/users/jniklaus/LEXTREME
export PYTHONPATH=. HF_DATASETS_CACHE=$SCRATCH/cache/datasets/${SLURM_ARRAY_TASK_ID} TRANSFORMERS_CACHE=$SCRATCH/cache/models

# Execute your program with the specified dataset, model, and GPU memory
python3 main.py -t ${dataset_name} -lmt ${model} -ld $SCRATCH/swiss_legal_data -gm ${gpu_memory} -los \${SLURM_ARRAY_TASK_ID} -rmo default -dmo reuse_dataset_if_exists

EOL

# Submit the job using sbatch
echo "Submitting job ${job_file}"
sbatch "${job_file}"
