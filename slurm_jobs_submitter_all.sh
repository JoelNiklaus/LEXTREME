#!/bin/bash

seeds="1,2,3"

# Set the different dataset names
dataset_names=(
    "swiss_judgment_prediction_xl_considerations"
    "swiss_judgment_prediction_xl_facts"
    "swiss_law_area_prediction_sub_area_considerations"
    "swiss_law_area_prediction_sub_area_facts"
    "swiss_law_area_prediction_considerations"
    "swiss_law_area_prediction_facts"
    "swiss_criticality_prediction_bge_considerations"
    "swiss_criticality_prediction_bge_facts"
    "swiss_criticality_prediction_citation_considerations"
    "swiss_criticality_prediction_citation_facts"
)

# Set the different models
models=(
    "xlm-roberta-base"
    "xlm-roberta-large"
    "joelito/legal-swiss-roberta-base"
    "joelito/legal-swiss-roberta-large"
    "joelito/legal-xlm-roberta-base"
    "joelito/legal-xlm-roberta-large"
)

mkdir -p jobfiles
mkdir -p slurm_logs

# Loop through the dataset names and models
for dataset_name in "${dataset_names[@]}"; do
  for model in "${models[@]}"; do

    # Invoke the create_job_file.sh script with the required arguments
    ./create_job_file.sh "$dataset_name" "$model" "$seeds"

  done
done

