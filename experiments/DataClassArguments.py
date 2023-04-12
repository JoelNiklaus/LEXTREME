#!/usr/bin/env python
# coding=utf-8


import os
import sys
from dataclasses import dataclass, field
from typing import Optional

path_to_nlp_scripts = os.path.join(os.path.dirname(__file__), '../utils/')
sys.path.append(path_to_nlp_scripts)

from utilities import get_meta_infos

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # Only for Politmonitor
    affair_text_scope: str = field(
        # default='ch',
        metadata={
            "help": "Specify which kanton you would like to include in the dataset."
        },
    )
    # Only for Politmonitor
    title: bool = field(
        # default=True,
        metadata={
            "help": "Specify if you want to include the title in the training."
        },
    )
    # Only for Politmonitor
    text: bool = field(
        # default=True,
        metadata={
            "help": "Specify if you want to include the text in the training."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_segments: Optional[int] = field(
        default=8,
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
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
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
    language: Optional[str] = field(
        default=None,
        metadata={
            "help": "For choosin the language "
                    "value if set."
        },
    )
    running_mode: Optional[str] = field(
        default='default',
        metadata={
            "help": "If set to 'experimental' only a small portion of the original dataset will be used for fast "
                    "experiments"
        },
    )
    finetuning_task: Optional[str] = field(
        default='finetuning_task',
        metadata={
            "help": "Name of the finetuning task"
        },
    )
    download_mode: Optional[str] = field(
        default='reuse_cache_if_exists',
        metadata={
            "help": "Name of the finetuning task"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataset_cache_dir: str = field(
        default=None,
        metadata={
            "help": "Specify the directory you want to cache your datasets."
        },
    )
    log_directory: str = field(
        default=None,
        metadata={
            "help": "Specify the directory where you want to save your logs."
        },
    )
    disable_caching: bool = field(
        default=False,
        metadata={
            "help": "Specify if you want to dsable caching on a global scale for the huggingface dataset objects with "
                    "disable_caching()."
        },
    )
    add_oversampling: bool = field(
        default=False,
        metadata={
            "help": "Specify if you want to add oversampling. This can only be done for SLTC tasks."
        },
    )

    server_ip: Optional[str] = field(
        default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(
        default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    hierarchical: bool = field(
        default=False, metadata={"help": "Whether to use a hierarchical variant or not"}
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
        default='./datasets_cache_dir',
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={
            "help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use. It can be a branch name, a tag name, or a commit id, "
                    "since we use a git-based system for storing models and other artifacts on huggingface.co, "
                    "so revision can be any identifier allowed by git."
        }
    )
    do_hyperparameter_search: bool = field(
        default=False,
        metadata={
            "help": "Specify if you want to apply hyperparameter search."
        },
    )


def get_default_values():
    meta_infos = get_meta_infos()
    task_default_arguments = meta_infos["task_default_arguments"]

    for task, languages in meta_infos["task_language_mapping"].items():
        if task in task_default_arguments.keys():
            if len(languages) == 1:
                language = languages[0]
            else:
                language = "all"
            task_default_arguments[task]["DataTrainingArguments"]["language"] = language
    return task_default_arguments
