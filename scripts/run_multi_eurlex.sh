value="$(cat running_mode.txt)"
RUNNING_MODE=$value

GPU_NUMBER=0
LOWER_CASE='True'
BATCH_SIZE=30
ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=20
LEARNING_RATE=1e-5



model_input="models_to_be_used.txt"
while IFS= read -r line
  MODEL_NAME="$line"
  
  do
    
    lang_input="languages_eur_lex.txt"
    while IFS= read -r line
    LANGUAGE="$line"
    TASK=${LANGUAGE}'_multi_eurlex'
    
    do

      CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python ../experiments/run_multi_eurlex.py --model_name_or_path ${MODEL_NAME} --do_lower_case ${LOWER_CASE}  --output_dir logs/${TASK}/${MODEL_NAME}/seed_1 --do_train --do_eval --do_pred --overwrite_output_dir --load_best_model_at_end --metric_for_best_model micro-f1 --greater_is_better True --evaluation_strategy epoch --save_strategy epoch --save_total_limit 5 --num_train_epochs ${NUM_TRAIN_EPOCHS} --learning_rate ${LEARNING_RATE} --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} --seed 1 --fp16 --fp16_full_eval --gradient_accumulation_steps ${ACCUMULATION_STEPS} --eval_accumulation_steps ${ACCUMULATION_STEPS} --language ${LANGUAGE} --running_mode ${RUNNING_MODE}

    done < "$lang_input"

  
  done < "$model_input"





python ../statistics/compute_avg_scores.py --dataset ${TASK}


