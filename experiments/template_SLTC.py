#!/usr/bin/env python
# coding=utf-8


import glob
import logging
import os
import shutil
import sys
from dataclasses import replace
import wandb

import transformers
from datasets import disable_caching
from datasets import utils
from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint

from DataClassArguments import DataTrainingArguments, ModelArguments, get_default_values
from helper import compute_metrics_multi_class, make_predictions_multi_class, config_wandb, \
    generate_Model_Tokenizer_for_SequenceClassification, preprocess_function, add_oversampling_to_multiclass_dataset, \
    get_data, get_label_list_from_sltc_tasks, model_is_multilingual, preprocess_datasets

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    default_values = get_default_values()

    if data_args.finetuning_task in default_values.keys():
        for argument_type in default_values[data_args.finetuning_task].keys():
            if argument_type == "DataTrainingArguments":
                for field, value in default_values[data_args.finetuning_task][argument_type].items():
                    if field == "language":
                        if model_is_multilingual(model_args.model_name_or_path) and data_args.language is None:
                            para = {field: value}
                            data_args = replace(data_args, **para)
                        elif data_args.language is None:
                            para = {field: value}
                            data_args = replace(data_args, **para)
                    else:
                        para = {field: value}
                        data_args = replace(data_args, **para)
            elif argument_type == "ModelArguments":
                for field, value in default_values[data_args.finetuning_task][argument_type].items():
                    para = {field: value}
                    model_args = replace(model_args, **para)

    if data_args.disable_caching:
        disable_caching()

    if data_args.finetuning_task in ["swiss_criticality_prediction_bge_facts",
                                     "swiss_criticality_prediction_bge_considerations",
                                     "swiss_criticality_prediction_citation_facts",
                                     "swiss_criticality_prediction_citation_considerations"]:
        os.environ["WANDB_API_KEY"] = "6f9fce3d2b3e41f8880b3e0b094e16ec9d030315"  # wandb rstern token

    config_wandb(model_args=model_args, data_args=data_args, training_args=training_args)

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Fix boolean parameter
    if model_args.do_lower_case == 'False' or not model_args.do_lower_case:
        model_args.do_lower_case = False
        'Tokenizer do_lower_case False'
    else:
        model_args.do_lower_case = True

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset, eval_dataset, predict_dataset = get_data(training_args, data_args)

    if data_args.finetuning_task in ["swiss_criticality_prediction_bge_facts",
                                     "swiss_criticality_prediction_bge_considerations",
                                     "swiss_criticality_prediction_citation_facts",
                                     "swiss_criticality_prediction_citation_considerations"]:
        train_dataset = preprocess_datasets(train_dataset, data_args.finetuning_task)
        eval_dataset = preprocess_datasets(eval_dataset, data_args.finetuning_task)
        predict_dataset = preprocess_datasets(predict_dataset, data_args.finetuning_task)

    # Labels
    label_list = get_label_list_from_sltc_tasks(train_dataset, eval_dataset, predict_dataset)
    num_labels = len(label_list)

    label2id = dict()
    id2label = dict()

    for n, l in enumerate(label_list):
        label2id[l] = n
        id2label[n] = l

    # NOTE: This is not optimized for multiclass classification
    if not (not (training_args.do_train == True) or not data_args.add_oversampling):
        logger.info("Oversampling the minority class")
        train_dataset = add_oversampling_to_multiclass_dataset(train_dataset=train_dataset, id2label=id2label,
                                                               data_args=data_args)

    model, tokenizer, config = generate_Model_Tokenizer_for_SequenceClassification(model_args=model_args,
                                                                                   data_args=data_args,
                                                                                   num_labels=num_labels)

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                lambda x: preprocess_function(x, tokenizer, model_args, data_args),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                lambda x: preprocess_function(x, tokenizer, model_args, data_args),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                lambda x: preprocess_function(x, tokenizer, model_args, data_args),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    training_args.metric_for_best_model = "eval_loss"
    training_args.evaluation_strategy = IntervalStrategy.EPOCH
    training_args.logging_strategy = IntervalStrategy.EPOCH

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics_multi_class,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        wandb.log(metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        wandb.log(metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        make_predictions_multi_class(trainer=trainer, training_args=training_args, data_args=data_args,
                                     predict_dataset=predict_dataset, id2label=id2label, name_of_input_field="input")

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
