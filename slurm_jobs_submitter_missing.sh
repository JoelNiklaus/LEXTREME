#!/bin/bash

my_models="microsoft/mdeberta-v3-base xlm-roberta-large joelito/legal-swiss-roberta-large google/mt5-base"

# Read the CSV file and loop through the lines
while IFS=$'\t' read -r dataset_name model language seeds revision responsible gpu; do
    # Skip the header line
    if [[ "$dataset_name" == "finetuning_task" ]]; then
        continue
    fi

    # skip the models that are not in my_models
    if ! [[ " $my_models " =~ " $model " ]]; then
      continue
    fi

    # skip the datasets that are not working
    if [[ "$dataset_name" == "swiss_law_area_prediction_civil_considerations" ]]; then
    continue
fi


    # Invoke the create_job_file.sh script with the required arguments
    ./create_job_file.sh "$dataset_name" "$model" "$seeds"

done < utils/results/completeness_report.tsv
