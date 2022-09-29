#!/usr/bin/env python
# coding=utf-8


import logging
import os
import random
import sys
import re
from dataclasses import dataclass, field
from typing import Optional
import json as js

import datasets
from helper import compute_metrics_multi_class, make_predictions_multi_class, config_wandb, generate_Model_Tokenizer_for_SequenceClassification, split_into_segments
import pandas as pd
from datasets import load_dataset
import numpy as np
import glob
import shutil
from models.hierbert import HierarchicalBert
from torch import nn


import transformers
from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_segments: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum number of segments (paragraphs) to be considered. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seg_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    language:Optional[str] = field(
        default='el',
        metadata={
            "help": "For choosin the language "
            "value if set."
        },
    )
    running_mode:Optional[str] = field(
        default='default',
        metadata={
            "help": "If set to 'experimental' only a small portion of the original dataset will be used for fast experiments"
        },
    )
    finetuning_task:Optional[str] = field(
        default='greek_legal_code_volume',
        metadata={
            "help": "Name of the finetuning task"
        },
    )

    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    hierarchical: bool = field(
        default=True, metadata={"help": "Whether to use a hierarchical variant or not"}
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    config_wandb(model_args=model_args, data_args=data_args,training_args=training_args)


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
    datasets.utils.logging.set_verbosity(log_level)
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

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading eurlex dataset from the hub.
    


    if training_args.do_train:
        train_dataset = load_dataset("joelito/lextreme",data_args.finetuning_task,split='train', cache_dir=model_args.cache_dir)
        

    if training_args.do_eval:
        eval_dataset = load_dataset("joelito/lextreme",data_args.finetuning_task,split='validation', cache_dir=model_args.cache_dir)

    if training_args.do_predict:
        predict_dataset = load_dataset("joelito/lextreme",data_args.finetuning_task,split='test', cache_dir=model_args.cache_dir)

    
    # Labels
    label_list = train_dataset.features['label'].names

    num_labels = len(label_list)

    label2id = dict()
    id2label = dict()

    for n,l in enumerate(label_list):
        label2id[l]=n
        id2label[n]=l

    if data_args.running_mode=='experimental':
        data_args.max_train_samples=1000
        data_args.max_eval_samples=200
        data_args.max_predict_samples=100

    model, tokenizer, config = generate_Model_Tokenizer_for_SequenceClassification(model_args=model_args, data_args=data_args, num_labels=num_labels)
    
    if model_args.hierarchical:
        # Hack the classifier encoder to use hierarchical BERT
        if config.model_type in ['bert','deberta-v2','distilbert']:
            if config.model_type == 'bert':
                segment_encoder = model.bert
            elif config.model_type =='distilbert':
                segment_encoder = model.distilbert
            elif config.model_type =='deberta-v2':
                segment_encoder = model.deberta
            model_encoder = HierarchicalBert(encoder=segment_encoder,
                                             max_segments=data_args.max_segments,
                                             max_segment_length=data_args.max_seg_length)
            if config.model_type == 'bert':
                model.bert = model_encoder
            elif config.model_type == 'distilbert':
                model.distilbert = model_encoder
            elif config.model_type == 'deberta-v2':
                model.deberta = model_encoder
            else:
                raise NotImplementedError(f"{config.model_type} is no supported yet!")

        elif config.model_type in ['roberta','xlm-roberta']:
            model_encoder = HierarchicalBert(encoder=model.roberta, 
                                            max_segments=data_args.max_segments,
                                            max_segment_length=data_args.max_seg_length)
            model.roberta = model_encoder
            # Build a new classification layer, as well
            dense = nn.Linear(config.hidden_size, config.hidden_size)
            dense.load_state_dict(model.classifier.dense.state_dict())  # load weights
            dropout = nn.Dropout(config.hidden_dropout_prob).to(model.device)
            out_proj = nn.Linear(config.hidden_size, config.num_labels).to(model.device)
            out_proj.load_state_dict(model.classifier.out_proj.state_dict())  # load weights
            model.classifier = nn.Sequential(dense, dropout, out_proj).to(model.device)

        elif config.model_type in ['longformer', 'big_bird']:
            pass
        else:
            raise NotImplementedError(f"{config.model_type} is no supported yet!")

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def append_zero_segments(case_encodings, pad_token_id):
        """appends a list of zero segments to the encodings to make up for missing segments"""
        return case_encodings + [[pad_token_id] * data_args.max_seg_length] * (
                data_args.max_segments - len(case_encodings))

    def preprocess_function(batch):
    
        pad_id = tokenizer.pad_token_id

        if model_args.hierarchical:
            batch['segments'] = []

            tokenized = tokenizer(batch["input"], padding=padding, truncation=True,
                                    max_length=data_args.max_segments * data_args.max_seg_length,
                                    add_special_tokens=False)  # prevent it from adding the cls and sep tokens twice
            for ids in tokenized['input_ids']:
                # convert ids to tokens and then back to strings
                id_blocks = [ids[i:i + data_args.max_seg_length] for i in range(0, len(ids), data_args.max_seg_length) if
                                ids[i] != pad_id]  # remove blocks containing only ids
                id_blocks[-1] = [id for id in id_blocks[-1] if
                                    id != pad_id]  # remove remaining pad_tokens_ids from the last block
                token_blocks = [tokenizer.convert_ids_to_tokens(ids) for ids in id_blocks]
                string_blocks = [tokenizer.convert_tokens_to_string(tokens) for tokens in token_blocks]
                batch['segments'].append(string_blocks)
                
            
            # Tokenize the text
            
            tokenized = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
            for case in batch['segments']:
                case_encodings = tokenizer(case[:data_args.max_segments], padding=padding, truncation=True,
                                        max_length=data_args.max_seg_length, return_token_type_ids=True)
                tokenized['input_ids'].append(append_zero_segments(case_encodings['input_ids'], pad_id))
                tokenized['attention_mask'].append(append_zero_segments(case_encodings['attention_mask'], 0))
                tokenized['token_type_ids'].append(append_zero_segments(case_encodings['token_type_ids'], 0))

            
            del batch['segments']

        else:
            tokenized = tokenizer(
                batch["input"],
            padding=padding,
                max_length=512, #Otherwise it would through an error, since the default value is too high, because per default we use the hierarchical model
            truncation=True,
        )

        return tokenized

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
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
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
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
    training_args.metric_for_best_model = "mcc"
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
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        make_predictions_multi_class(trainer=trainer,training_args=training_args,data_args=data_args,predict_dataset=predict_dataset,id2label=id2label,name_of_input_field="input")


    
    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()