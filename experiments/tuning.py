#!/usr/bin/env python
# coding=utf-8

import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
)
import logging
import os
import random
import sys
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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config_wandb(model_args=model_args, data_args=data_args, training_args=training_args, project_name='sltc_params_tuning')

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

    ##################################################################################
    # Changes:

    logger.info(f"Training arguments {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")

    # Set hyperparam batch_size:

    # models: "xlm-roberta-base","microsoft/mdeberta-v3-base" "joelito/legal-xlm-roberta-large" "joelito/legal-xlm-roberta-base"

    if model_args.model_name_or_path == "xlm-roberta-large":
        training_args.per_device_eval_batch_size = 2
        training_args.per_device_train_batch_size = 2
    elif model_args.model_name_or_path == 'joelito/legal-xlm-roberta-large':
        training_args.per_device_eval_batch_size = 2
        training_args.per_device_train_batch_size = 2
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

    if data_args.running_mode == "experimental":
        experimental_samples = True
    else:
        experimental_samples = False


    feature_col = 'facts'  # must be either facts, considerations (or in future judgment)
    # remove_non_critical = false
    # filter_attribute_value = social_law  # must be a region (...) or a legal area (...)

    if data_args.finetuning_task == 'citation_prediction':
        task = 'cit'
        label = 'laws'
    else:
        task = 'crit'
        label = 'bge_label'
    ds_dict = {'train': get_dataset(experimental_samples, training_args.do_train, 'train', task),
               'validation': get_dataset(experimental_samples, training_args.do_eval, 'validation', task),
               'test': get_dataset(experimental_samples, training_args.do_predict, 'test', task)}

    for k in ds_dict:
        ds_dict[k] = remove_unused_features(ds_dict[k], feature_col, label)
        ds_dict[k] = rename_features(ds_dict[k], feature_col, label)
        ds_dict[k] = ds_dict[k].filter(filter_by_length)

    ds_dict, label_list = cast_label_to_labelclass(ds_dict, label)  # Do this for all datasets at the same time

    train_dataset = ds_dict['train']
    eval_dataset = ds_dict['validation']
    predict_dataset = ds_dict['test']

    logger.info(train_dataset)
    num_labels = len(label_list)

    # End changes
    ######################################################################################

    label2id = dict()
    id2label = dict()

    for n, l in enumerate(label_list):
        label2id[l] = n
        id2label[n] = l

    # NOTE: This is not optimized for multiclass classification
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

    """
    training_args = TrainingArguments(
        output_dir=".",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
    )
    """

    trainer = Trainer(
        model_init=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics_multi_class,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    tune_config = {
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_acc",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [2, 4, 8, 16],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_loss", "epoch", "training_iteration"],
    )

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=10,
        resources_per_trial={"cpu": 1, "gpu": 1},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop=None,
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    )


if __name__ == "__main__":
    ray.init()
    main()
