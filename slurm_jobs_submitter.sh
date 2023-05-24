#!/bin/bash

# IMPORTANT: Run conda init and conda activate lextreme before running this script

my_models="joelito/legal-swiss-roberta-base joelito/legal-swiss-roberta-large"
my_models="joelito/legal-xlm-longformer-base"
my_models="joelito/legal-german-roberta-base joelito/legal-german-roberta-large joelito/legal-french-roberta-base joelito/legal-french-roberta-large joelito/legal-italian-roberta-base joelito/legal-italian-roberta-large"
my_models="distilbert-base-multilingual-cased microsoft/Multilingual-MiniLM-L12-H384 microsoft/mdeberta-v3-base xlm-roberta-base xlm-roberta-large joelito/legal-swiss-longformer-base joelito/legal-swiss-roberta-base joelito/legal-swiss-roberta-large google/mt5-small google/mt5-base google/mt5-large facebook/xmod-base ZurichNLP/swissbert ZurichNLP/swissbert-xlm-vocab"
my_datasets="greek_legal_code_chapter greek_legal_code_subject greek_legal_code_volume swiss_judgment_prediction online_terms_of_service_unfairness_levels online_terms_of_service_clause_topics covid19_emergency_event multi_eurlex_level_1 multi_eurlex_level_2 multi_eurlex_level_3 greek_legal_ner legalnero lener_br mapa_coarse mapa_fine"
my_datasets="greek_legal_code_chapter greek_legal_code_volume swiss_judgment_prediction online_terms_of_service_unfairness_levels online_terms_of_service_clause_topics covid19_emergency_event multi_eurlex_level_1 multi_eurlex_level_2 multi_eurlex_level_3 greek_legal_ner legalnero lener_br mapa_coarse mapa_fine"
my_datasets="swiss_judgment_prediction_xl_facts swiss_judgment_prediction_xl_considerations swiss_law_area_prediction_facts swiss_law_area_prediction_considerations swiss_law_area_prediction_sub_area_facts swiss_law_area_prediction_sub_area_considerations"

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

    # skip the datasets that are not in my_datasets
    if ! [[ " $my_datasets " =~ " $dataset_name " ]]; then
      continue
    fi

    # Invoke the create_job_file.sh script with the required arguments
    ./create_job_file.sh "$dataset_name" "$model" "$revision" "$seeds"

done < utils/results/swiss-legal-data/neurips2023/completeness_report.tsv
#done < utils/results/lextreme/paper_results/completeness_report.tsv
