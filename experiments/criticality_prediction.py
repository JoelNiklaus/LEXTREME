#!/usr/bin/env python
# coding=utf-8


import logging
import os
import random
import sys

import wandb
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from dataclasses import replace

from helper import compute_metrics_multi_class, make_predictions_multi_class, config_wandb, \
    generate_Model_Tokenizer_for_SequenceClassification, preprocess_function, add_oversampling_to_multiclass_dataset, \
    get_data, model_is_multilingual
from helper_cp import get_dataset, remove_unused_features, rename_features, filter_by_length,\
    add_oversampling_to_multiclass_dataset, cast_label_to_labelclass
from datasets import utils
import glob
import shutil
import datasets
from models.hierbert import HierarchicalBert
from torch import nn
from datasets import disable_caching
from datasets import Dataset, concatenate_datasets, load_dataset, load_metric

import transformers
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
from DataClassArguments import DataTrainingArguments, ModelArguments, get_default_values
from transformers.trainer_utils import get_last_checkpoint

disable_caching()

logger = logging.getLogger(__name__)

os.environ["WANDB_API_KEY"] = "6f9fce3d2b3e41f8880b3e0b094e16ec9d030315"

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

    config_wandb(model_args=model_args, data_args=data_args, training_args=training_args, project_name='scp-test')

    # Start Changes Criticality
    ##################################################################################

    # set up different experiments
    feature_col = 'facts'  # must be either facts, considerations or rulings
    label = 'citation_label'  # must be either bge_label or citation_label
    only_critical = True  # Only for citation_label

    # training_args.fp16

    # Set hyperparameter batch_size:
    if model_args.model_name_or_path == "xlm-roberta-large":
        training_args.per_device_eval_batch_size = 2
        training_args.per_device_train_batch_size = 2
    elif model_args.model_name_or_path == 'joelito/legal-xlm-roberta-large':
        training_args.per_device_eval_batch_size = 4
        training_args.per_device_train_batch_size = 4
    elif model_args.model_name_or_path == "xlm-roberta-base":
        training_args.per_device_eval_batch_size = 4
        training_args.per_device_train_batch_size = 4
    elif model_args.model_name_or_path == 'joelito/legal-xlm-roberta-base':
        training_args.per_device_eval_batch_size = 4
        training_args.per_device_train_batch_size = 4
    elif model_args.model_name_or_path == 'microsoft/mdeberta-v3-base':
        training_args.per_device_eval_batch_size = 8
        training_args.per_device_train_batch_size = 8
    else:
        exit()

    # Set hyperparameter learning_rate

    # Set hyperparameter weight_decay to avoid catastrophic forgetting

    # Set hyperparameter num_train_epochs

    # log all arguments
    logger.info(f"Training arguments {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    # get datasets
    ds_dict = {'train': get_dataset(data_args.running_mode, training_args.do_train, 'train', data_args.finetuning_task),
               'validation': get_dataset(data_args.running_mode, training_args.do_eval, 'validation', data_args.finetuning_task),
               'test': get_dataset(data_args.running_mode, training_args.do_predict, 'test', data_args.finetuning_task)}

    for k in ds_dict:
        ds_dict[k] = remove_unused_features(ds_dict[k], feature_col, label)
        ds_dict[k] = rename_features(ds_dict[k], feature_col, label)
        ds_dict[k] = ds_dict[k].filter(filter_by_length)
        if only_critical and label == 'citation_label':  # only use critical cases
            ds_dict[k] = ds_dict[k].filter(lambda row: row['label'].startswith("critical"))

    ds_dict, label_list = cast_label_to_labelclass(ds_dict, label)  # Do this for all datasets at the same time
    num_labels = len(label_list)

    train_dataset = ds_dict['train']
    eval_dataset = ds_dict['validation']
    predict_dataset = ds_dict['test']

    logger.info(train_dataset)
    logger.info(eval_dataset)
    logger.info(predict_dataset)

    # End changes Criticality
    ######################################################################################

    label2id = dict()
    id2label = dict()

    for n, l in enumerate(label_list):
        label2id[l] = n
        id2label[n] = l

    if training_args.do_train:
        logger.info("Oversampling the minority class")
        train_dataset = add_oversampling_to_multiclass_dataset(train_dataset=train_dataset, id2label=id2label,
                                                               data_args=data_args)
    logger.info(train_dataset)
    logger.info(train_dataset.features)

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

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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
        wandb.log("train", metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        wandb.log("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        langs = train_dataset['language'] + eval_dataset['language'] + predict_dataset['language']
        langs = sorted(list(set(langs)))

        make_predictions_multi_class(trainer=trainer, training_args=training_args, data_args=data_args,
                                     predict_dataset=predict_dataset, id2label=id2label, name_of_input_field="input",
                                     list_of_languages=langs)

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
