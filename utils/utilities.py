# Original
import random

import arrow
from collections import defaultdict
import json as js
import os
from pathlib import Path
import datetime
import re
from ast import literal_eval
import setproctitle
import itertools
from itertools import cycle
from multiprocessing import Pool
import torch
from time import sleep

# In the beginning we have the configurations
optimal_batch_sizes = {
    # e.g. RTX 2080 Ti or GTX 1080 Ti
    # TODO test sizes here
    11: {
        # untested
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8, 8192: 8},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 16, 512: 8, 1024: 4, 2048: 2, 4096: 1, 8192: 1},
        # 8192 untested
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        # 8192 untested
        'monolingual_base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # model is too large
        'monolingual_large': {256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0},
        # 8192 untested
        'xlm-roberta-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # 8192 untested
        'google/mt5-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # lower batch sizes because not possible with fp16
        # model is too large
        'microsoft/mdeberta-v3-base': {256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0},
        # model is too large
        'google/mt5-large': {256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0},
        # model is too large
        'xlm-roberta-large': {256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0},
    },
    # TODO test sizes here
    # e.g. P100
    16: {
        # untested
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8, 8192: 8},
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        # 8192 untested
        'monolingual_base': {256: 32, 512: 32, 1024: 8, 2048: 4, 4096: 4, 8192: 4},
        # untested
        'monolingual_large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # untested
        'xlm-roberta-base': {256: 32, 512: 32, 1024: 16, 2048: 4, 4096: 4, 8192: 4},
        # untested
        'google/mt5-base': {256: 32, 512: 32, 1024: 16, 2048: 4, 4096: 4, 8192: 4},
        # lower batch sizes because not possible with fp16
        # untested
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # untested
        'google/mt5-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # untested
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
    },
    # e.g. RTX 3090
    24: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 8192 untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8, 8192: 8},
        # 8192 untested
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        # 8192 untested
        'monolingual_base': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8, 8192: 8},
        # 8192 untested
        'monolingual_large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # 8192 untested
        'xlm-roberta-base': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8, 8192: 8},
        # 8192 untested
        'google/mt5-base': {256: 64, 512: 32, 1024: 16, 2048: 4, 4096: 4, 8192: 4},
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        'google/mt5-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
    },
    # e.g. V100
    32: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 8192 untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 8192 untested
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        # 8192 untested
        'monolingual_base': {256: 64, 512: 32, 1024: 32, 2048: 8, 4096: 8, 8192: 8},
        # 8192 untested
        'monolingual_large': {256: 8, 512: 4, 1024: 4, 2048: 2, 4096: 2, 8192: 2},
        # 8192 untested
        'xlm-roberta-base': {256: 32, 512: 16, 1024: 16, 2048: 8, 4096: 4, 8192: 4},
        # 8192 untested
        'google/mt5-base': {256: 32, 512: 16, 1024: 16, 2048: 8, 4096: 4, 8192: 4},
        # lower batch sizes because not possible with fp16
        # 8192 untested
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 2},
        # 8192 untested
        'google/mt5-large': {256: 8, 512: 4, 1024: 4, 2048: 2, 4096: 2, 8192: 2},
        # 8192 untested
        'xlm-roberta-large': {256: 8, 512: 4, 1024: 4, 2048: 2, 4096: 2, 8192: 2},
    },
    # TODO test sizes here
    # e.g. A6000
    48: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 64, 8192: 64},
        # 8192 untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32, 8192: 32},
        # 8192 untested
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        # 8192 untested
        'monolingual_base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 8192 untested
        'monolingual_large': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 4, 8192: 4},
        # 8192 untested
        'xlm-roberta-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 8192 untested
        'google/mt5-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # lower batch sizes because not possible with fp16
        # bf16; 8192 untested
        'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8, 8192: 18},
        # 'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # fp32
        # fp16
        'google/mt5-large': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 4, 8192: 4},
        # fp16
        'xlm-roberta-large': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 4, 8192: 4},
    },
    # e.g. A100
    80: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 64, 8192: 64},
        # fp16 ; 8192 untested
        # 'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},  # fp32
        # 8192 untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32, 8192: 32},
        # fp16 ; 8192 untested
        # 'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},  # fp32
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        # 8192 untested
        'monolingual_base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 8192 untested
        'monolingual_large': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8, 8192: 8},
        # fp16
        'xlm-roberta-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # fp16; 8192 untested
        'google/mt5-base': {256: 64, 512: 32, 1024: 8, 2048: 16, 4096: 8, 8192: 4},
        # 'xlm-roberta-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16},  # fp32
        # lower batch sizes because not possible with fp16
        # bf16; 8192 untested
        'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16, 8192: 16},
        # 'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # fp32
        # fp16
        'google/mt5-large': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 4, 8192: 2},
        # fp16
        'xlm-roberta-large': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8, 8192: 4},
        # 'xlm-roberta-large': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 4}, # fp32
    },
}



def find_optimal_learning_rate(model_name_or_path: str):
    if model_name_or_path in ['google/mt5-small', 'google/mt5-base', 'google/mt5-large']:
        return "1e-4"
    else:
        return "1e-5"


def make_boolean(value):
    value_capitalized = str(value).lower().title()
    if literal_eval(value_capitalized) in [True, False, None]:
        return literal_eval(str(value).title())
    else:
        return value


def remove_old_files(path_to_directory, days=-30):
    # keep files for 30 days to make sure that this works also in a hpc environment
    criticalTime = arrow.now().shift().shift(days=days)
    print(
        f"Removing old temporary bash scripts (older than {days} days: before {criticalTime})")

    for file in Path(path_to_directory).glob('*'):
        if file.is_file():
            file_time_modified = arrow.get(file.stat().st_mtime)
            if file_time_modified < criticalTime:
                try:
                    os.remove(file)
                except OSError:
                    pass  # we don't care if the file is not there


def get_meta_infos():
    path_to_json_file = os.path.join(
        os.path.dirname(__file__), 'meta_infos.json')
    with open(path_to_json_file, 'r') as f:
        meta_infos = js.load(f)

    language_models = meta_infos["language_models"]

    # Creating a dictionary to look up the language for each language model
    model_language_lookup_table = dict()
    for k, v in language_models.items():
        if type(v) == dict:
            for language, info in v.items():
                if language == "multilingual":
                    language = "all"
                for size, models in info.items():
                    for m in models:
                        model_language_lookup_table[m] = language

    meta_infos["model_language_lookup_table"] = model_language_lookup_table

    code_task_mapping = defaultdict(list)

    for k, v in meta_infos["task_type_mapping"].items():
        code_task_mapping[v].append(k)

    code_task_mapping = dict(code_task_mapping)

    meta_infos["code_task_mapping"] = code_task_mapping

    task_requires_hierarchical_per_default = dict()
    for task, infos in meta_infos["task_default_arguments"].items():
        if infos["ModelArguments"]["hierarchical"]:
            task_requires_hierarchical_per_default[task] = "yes"
        if not infos["ModelArguments"]["hierarchical"]:
            task_requires_hierarchical_per_default[task] = "no"

    meta_infos[
        "task_requires_hierarchical_per_default"] = task_requires_hierarchical_per_default

    model_size_mapping = dict()
    for model_type, model_info in meta_infos['language_models'].items():
        for language, info in model_info.items():
            for size, models in info.items():
                for m in models:
                    model_size_mapping[m] = size
    meta_infos["model_size_mapping"] = model_size_mapping

    model_type_mapping = dict()
    for model_type, infos in meta_infos['language_models'].items():
        for language, size_and_models in infos.items():
            for size, models in size_and_models.items():
                for m in models:
                    model_type_mapping[m] = model_type
    meta_infos["model_type_mapping"] = model_type_mapping

    _TYPES = sorted(list(set(meta_infos["model_type_mapping"].values())))
    _LANGUAGES = sorted(
        list(set(meta_infos['model_language_lookup_table'].values())))
    _SIZES = sorted(list(set(meta_infos['model_size_mapping'].values())))

    meta_infos['_TYPES'] = _TYPES
    meta_infos['_LANGUAGES'] = _LANGUAGES
    meta_infos['_SIZES'] = _SIZES

    all_available_models = []
    for domain, model_infos in meta_infos['language_models'].items():
        for language, size_and_models in model_infos.items():
            for models in size_and_models.values():
                for _name_or_path in models:
                    all_available_models.append(_name_or_path)

    meta_infos["all_available_models"] = all_available_models

    return meta_infos


meta_infos = get_meta_infos()

# TODO get this directly from the LEXTREME huggingface dataset loader
# SLTC: Single Class Text Classification
# MLTC: Multi Class Text Classification
# NER: Named Entity Recognition
task_type_mapping = meta_infos["task_type_mapping"]

max_sequence_lengths = dict()
for task in meta_infos["task_default_arguments"].keys():
    if "max_segments" in meta_infos["task_default_arguments"][task][
        "DataTrainingArguments"].keys() and "max_seg_length" in meta_infos["task_default_arguments"][task][
        "DataTrainingArguments"]:
        max_sequence_lengths[task] = meta_infos["task_default_arguments"][task]["DataTrainingArguments"][
                                         "max_segments"] * \
                                     meta_infos["task_default_arguments"][task]["DataTrainingArguments"][
                                         "max_seg_length"]
    else:
        max_sequence_lengths[task] = meta_infos["task_default_arguments"][task]["DataTrainingArguments"][
            "max_seq_length"]


def get_hierarchical(task):
    if meta_infos["task_requires_hierarchical_per_default"][task] == "yes":
        return True
    else:
        return False


def get_python_file_for_task(task):
    if meta_infos["task_type_mapping"][task] == "NER":
        return f"template_NER.py --finetuning_task " + task
    elif meta_infos["task_type_mapping"][task] == "SLTC":
        return f"template_SLTC.py --finetuning_task " + task
    elif meta_infos["task_type_mapping"][task] == "MLTC":
        return f"template_MLTC.py --finetuning_task " + task


def get_optimal_batch_size(language_model: str, task: str, gpu_memory, total_batch_size=64):
    max_seq_len = max_sequence_lengths[task]
    try:
        batch_size_dict = optimal_batch_sizes[int(gpu_memory)][language_model]
    except:
        print(
            f"Did not find a batch size for the language model {language_model} and the gpu memory {gpu_memory}")
        if "joelito" in language_model:
            if "base" in language_model:
                print(
                    f"We assume that the language model {language_model} is based on xlm-roberta-base.")
                batch_size_dict = optimal_batch_sizes[int(
                    gpu_memory)]["xlm-roberta-base"]
            elif "large" in language_model:
                print(
                    f"We assume that the language model {language_model} is based on xlm-roberta-large.")
                batch_size_dict = optimal_batch_sizes[int(
                    gpu_memory)]["xlm-roberta-large"]
            else:
                raise ValueError(
                    "Did not find the language model in the dictionary.")
        else:
            print(f"The language model {language_model} will be considered a monolingual model. "
                  "Therefore, we revert to the default batch size.")
            if language_model not in meta_infos['model_size_mapping'].keys():
                batch_size_dict = optimal_batch_sizes[int(
                    gpu_memory)]["monolingual_base"]
            elif meta_infos['model_size_mapping'][language_model] != 'large':
                batch_size_dict = optimal_batch_sizes[int(
                    gpu_memory)]["monolingual_base"]
            else:
                batch_size_dict = optimal_batch_sizes[int(
                    gpu_memory)]["monolingual_large"]
    batch_size = None
    while batch_size is None:
        try:
            batch_size = batch_size_dict[max_seq_len]
        except KeyError:
            # if the sequence length is not in the dictionary, we try the next higher one
            max_seq_len *= 2
    accumulation_steps = total_batch_size // batch_size

    return batch_size, accumulation_steps


def get_strategies():
    pass


def get_default_number_of_training_epochs(task, model_name, running_mode, language=None):
    output_dict = dict()

    eval_steps = None
    logging_steps = None
    save_steps = None

    if meta_infos["model_language_lookup_table"][model_name] == "all":
        if 'multi_eurlex' in task:
            if language is None and running_mode == "default":
                # this dataset is so large, one epoch is enough to save compute
                # anyway, it starts overfitting for distilbert-base-multilingual-cased already at epoch 2 when training multilingually
                num_train_epochs = 1
                evaluation_strategy = "steps"
                logging_strategy = "steps"
                save_strategy = "steps"
                eval_steps = 1000
                logging_steps = 1000
                save_steps = 1000
            elif language in ['all', 'multilingual'] and running_mode == "default":
                # this dataset is so large, one epoch is enough to save compute
                # anyway, it starts overfitting for distilbert-base-multilingual-cased already at epoch 2 when training multilingually
                num_train_epochs = 1
                evaluation_strategy = "steps"
                logging_strategy = "steps"
                save_strategy = "steps"
                eval_steps = 1000
                logging_steps = 1000
                save_steps = 1000
            else:
                num_train_epochs = 50
                evaluation_strategy = "epoch"
                logging_strategy = "epoch"
                save_strategy = "epoch"
        elif task in ["swiss_judgment_prediction_xl_considerations", "swiss_judgment_prediction_xl_facts"]:
            if running_mode == "default":
                # this dataset is so large, one epoch is enough to save compute
                # anyway, it starts overfitting for distilbert-base-multilingual-cased already at epoch 2 when training multilingually
                num_train_epochs = 1
                evaluation_strategy = "steps"
                logging_strategy = "steps"
                save_strategy = "steps"
                eval_steps = 1000
                logging_steps = 1000
                save_steps = 1000
            else:
                num_train_epochs = 50
                evaluation_strategy = "epoch"
                logging_strategy = "epoch"
                save_strategy = "epoch"
        else:
            num_train_epochs = 50
            evaluation_strategy = "epoch"
            logging_strategy = "epoch"
            save_strategy = "epoch"

    else:
        num_train_epochs = 50
        evaluation_strategy = "epoch"
        logging_strategy = "epoch"
        save_strategy = "epoch"

    output_dict["num_train_epochs"] = num_train_epochs

    output_dict["evaluation_strategy"] = evaluation_strategy
    output_dict["logging_strategy"] = logging_strategy
    output_dict["save_strategy"] = save_strategy
    output_dict["eval_steps"] = eval_steps
    output_dict["logging_steps"] = logging_steps
    output_dict["save_steps"] = save_steps

    return output_dict


def generate_command_for_experiments(**data):
    time_now = datetime.datetime.now().isoformat()

    if data["hierarchical"] is None:
        data["hierarchical"] = get_hierarchical(data["task"])
    if data["learning_rate"] is None:
        data["learning_rate"] = find_optimal_learning_rate(data["model_name"])

    command_template = 'python3 ./experiments/{CODE} ' \
                       '--do_eval ' \
                       '--do_predict ' \
                       '--do_train ' \
                       '--download_mode {DOWNLOAD_MODE} ' \
                       '--eval_accumulation_steps {ACCUMULATION_STEPS} ' \
                       '--early_stopping_patience {EARLY_STOPPING_PATIENCE} ' \
                       '--evaluation_strategy {EVALUATION_STRATEGY} ' \
                       '--gradient_accumulation_steps {ACCUMULATION_STEPS} ' \
                       '--hierarchical {HIERARCHICAL} ' \
                       '--learning_rate {LEARNING_RATE} ' \
                       '--load_best_model_at_end ' \
                       '--log_directory {LOG_DIRECTORY} ' \
                       '--logging_strategy {LOGGING_STRATEGY} ' \
                       '--metric_for_best_model {METRIC_FOR_BEST_MODEL} ' \
                       '--model_name_or_path {MODEL_NAME} ' \
                       '--num_train_epochs {NUM_TRAIN_EPOCHS} ' \
                       '--overwrite_output_dir ' \
                       '--per_device_eval_batch_size {BATCH_SIZE} ' \
                       '--per_device_train_batch_size {BATCH_SIZE} ' \
                       '--preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} ' \
                       '--revision {REVISION} ' \
                       '--running_mode {RUNNING_MODE} ' \
                       '--save_strategy {SAVE_STRATEGY} ' \
                       '--save_total_limit 6 ' \
                       '--seed {SEED} ' \
                       '--do_fp16 {DO_FP16}'

    if data["dataset_cache_dir"] is not None:
        command_template = command_template + \
                           ' --dataset_cache_dir {DATASET_CACHE_DIR}'

    if data['greater_is_better'] is not None:
        command_template = command_template + \
                           ' --greater_is_better {GREATER_IS_BETTER} '

    if (data["language"] is not None):
        command_template = command_template + ' --language {LANGUAGE} '
        command_template = command_template + '--output_dir {LOG_DIRECTORY}/{TASK}/{' \
                                              'MODEL_NAME}/results_after_training_only_on_language_with_id__{' \
                                              'LANGUAGE}/seed_{SEED}'
    else:
        command_template = command_template + \
                           ' --output_dir {LOG_DIRECTORY}/{TASK}/{MODEL_NAME}/seed_{SEED} '

    command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} ' + command_template

    run_on_cpu = "gpu_number" not in data.keys() or not bool(
        re.search(r"\d", str(data["gpu_number"])))

    if run_on_cpu:
        # If no GPU available, we cannot make use of --fp16 --fp16_full_eval
        data["gpu_number"] = ""
    else:  # only when we have a GPU, we can run fp16 training
        if data["do_fp16"] == False:
            if data["model_name"] == "microsoft/mdeberta-v3-base":
                if int(data["gpu_memory"]) == 80:  # A100 also supports bf16
                    command_template += ' --bf16 --bf16_full_eval'
            elif 't5' not in data["model_name"]:
                # It seems like fp16 triggers eval/loss nan with t5: https://discuss.huggingface.co/t/t5-variants-return-training-loss-0-and-validation-loss-nan-while-fine-tuning/30839/5
                # --fp16_full_eval removed because they cause errors: transformers RuntimeError: expected scalar type Half but found Float
                # BUT, if the environment is set up correctly, also fp16_full_eval should work
                # We perceived some issues with xlm-roberta-base and
                if str(data["hierarchical"]).lower() == 'true':
                    # xlm-roberta-large. They returned a nan loss with fp16 in combination with hierarchical models
                    if not bool(re.search('(-large|ZurichNLP/swissbert)', data["model_name"])):
                        command_template += ' --fp16 --fp16_full_eval'
                    else:
                        command_template += ' --fp16 '
                else:
                    command_template += ' --fp16 --fp16_full_eval'
        elif data["do_fp16"] == True:
            command_template += ' --fp16 --fp16_full_eval'

    if 'logging_steps' in data.keys() and data['logging_steps'] is not None:
        command_template += ' --logging_steps ' + \
                            str(data["logging_steps"]) + ' '
    if 'eval_steps' in data.keys() and data['eval_steps'] is not None:
        command_template += ' --eval_steps ' + str(data["eval_steps"]) + ' '
    if 'save_steps' in data.keys() and data['save_steps'] is not None:
        command_template += ' --save_steps ' + str(data["save_steps"]) + ' '
    if data["lower_case"]:
        command_template = command_template + ' --do_lower_case '
    if data["weight_decay"] is not None:
        command_template = command_template + \
                           ' --weight_decay ' + str(data["weight_decay"])
    if data["warmup_ratio"] is not None:
        command_template = command_template + \
                           ' --warmup_ratio ' + str(data["warmup_ratio"])

    # mdeberta does not work with fp16 because it was trained with bf16
    # probably similar for MobileBERT: https://github.com/huggingface/transformers/issues/11327
    # For some reason microsoft/mdeberta-v3-base token classification returns eval_loss == NaN when using fp16

    final_command = command_template.format(GPU_NUMBER=data["gpu_number"],
                                            MODEL_NAME=data["model_name"],
                                            LOWER_CASE=data["lower_case"],
                                            TASK=data["task"], SEED=data["seed"],
                                            NUM_TRAIN_EPOCHS=data["num_train_epochs"],
                                            BATCH_SIZE=data["batch_size"],
                                            ACCUMULATION_STEPS=data["accumulation_steps"],
                                            LANGUAGE=data["language"],
                                            RUNNING_MODE=data["running_mode"],
                                            LEARNING_RATE=data["learning_rate"],
                                            CODE=data["code"],
                                            METRIC_FOR_BEST_MODEL=data["metric_for_best_model"],
                                            GREATER_IS_BETTER=data["greater_is_better"],
                                            DOWNLOAD_MODE=data["download_mode"],
                                            LOG_DIRECTORY=data["log_directory"],
                                            PREPROCESSING_NUM_WORKERS=data["preprocessing_num_workers"],
                                            DATASET_CACHE_DIR=data["dataset_cache_dir"],
                                            HIERARCHICAL=data["hierarchical"],
                                            EVALUATION_STRATEGY=data["evaluation_strategy"],
                                            LOGGING_STRATEGY=data["logging_strategy"],
                                            SAVE_STRATEGY=data["save_strategy"],
                                            REVISION=data["revision"],
                                            EARLY_STOPPING_PATIENCE=data["early_stopping_patience"],
                                            DO_FP16=data["do_fp16"]
                                            )

    file_name = './temporary_scripts/' + data["task"] + "_" + str(data["gpu_number"]) + "_" + str(
        data["seed"]) + "_" + str(data["model_name"]).replace('/', '_') + "_" + time_now + ".sh"

    with open(file_name, "w") as f:
        print(final_command, file=f)

    return file_name


def generate_command_for_hyperparameter_search(**data):
    time_now = datetime.datetime.now().isoformat()

    if data["hierarchical"] is None:
        data["hierarchical"] = get_hierarchical(data["task"])
    if data["learning_rate"] is None:
        data["learning_rate"] = find_optimal_learning_rate(data["model_name"])

    command_template = 'python3 ./experiments/{CODE} ' \
                       '--do_eval ' \
                       '--do_hyperparameter_search ' \
                       '--do_predict ' \
                       '--do_train ' \
                       '--download_mode {DOWNLOAD_MODE} ' \
                       '--evaluation_strategy {EVALUATION_STRATEGY} ' \
                       '--early_stopping_patience {EARLY_STOPPING_PATIENCE} ' \
                       '--hierarchical {HIERARCHICAL} ' \
                       '--learning_rate {LEARNING_RATE} ' \
                       '--log_directory {LOG_DIRECTORY} ' \
                       '--logging_strategy {LOGGING_STRATEGY} ' \
                       '--metric_for_best_model {METRIC_FOR_BEST_MODEL} ' \
                       '--model_name_or_path {MODEL_NAME} ' \
                       '--output_dir {LOG_DIRECTORY}/{TASK}/{MODEL_NAME} ' \
                       '--overwrite_output_dir ' \
                       '--preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} ' \
                       '--revision {REVISION} ' \
                       '--running_mode {RUNNING_MODE} ' \
                       '--save_strategy {SAVE_STRATEGY} ' \
                       '--search_type_method {SEARCH_TYPE_METHOD} ' \
                       '--early_stopping_patience {EARLY_STOPPING_PATIENCE}'

    if data["dataset_cache_dir"] is not None:
        command_template = command_template + \
                           ' --dataset_cache_dir {DATASET_CACHE_DIR}'

    if data['greater_is_better'] is not None:
        command_template = command_template + \
                           ' --greater_is_better {GREATER_IS_BETTER} '

    if data["language"] is not None:
        command_template = command_template + ' --language {LANGUAGE} '
    if data["lower_case"]:
        command_template = command_template + ' --do_lower_case '

    command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} ' + command_template

    run_on_cpu = "gpu_number" not in data.keys() or not bool(
        re.search(r"\d", str(data["gpu_number"])))

    if run_on_cpu:
        # If no GPU available, we cannot make use of --fp16 --fp16_full_eval
        data["gpu_number"] = ""
    else:  # only when we have a GPU, we can run fp16 training
        if data["model_name"] == "microsoft/mdeberta-v3-base":
            if int(data["gpu_memory"]) == 80:  # A100 also supports bf16
                command_template += ' --bf16 --bf16_full_eval'
        else:
            # --fp16_full_eval removed because they cause errors: transformers RuntimeError: expected scalar type Half but found Float
            # BUT, if the environment is set up correctly, also fp16_full_eval should work
            if str(data[
                       "hierarchical"]).lower() == 'true':  # We percieved some issues with xlm-roberta-base and
                # xlm-roberta-large. They returned a nan loss with fp16 in comnination with hierarchical models
                if bool(re.search('(xlm-roberta-base|xlm-roberta-large)', data["model_name"])) == False:
                    command_template += ' --fp16 --fp16_full_eval'
                else:
                    command_template += ' --fp16 '
            else:
                command_template += ' --fp16 --fp16_full_eval'

    final_command = command_template.format(ACCUMULATION_STEPS=data["accumulation_steps"],
                                            BATCH_SIZE=data["batch_size"],
                                            CODE=data["code"],
                                            DATASET_CACHE_DIR=data["dataset_cache_dir"],
                                            DOWNLOAD_MODE=data["download_mode"],
                                            GPU_NUMBER=data["gpu_number"],
                                            EVALUATION_STRATEGY=data["evaluation_strategy"],
                                            GREATER_IS_BETTER=data["greater_is_better"],
                                            HIERARCHICAL=data["hierarchical"],
                                            LANGUAGE=data["language"],
                                            LEARNING_RATE=data["learning_rate"],
                                            LOGGING_STRATEGY=data["logging_strategy"],
                                            LOG_DIRECTORY=data["log_directory"],
                                            LOWER_CASE=data["lower_case"],
                                            METRIC_FOR_BEST_MODEL=data["metric_for_best_model"],
                                            MODEL_NAME=data["model_name"],
                                            NUM_TRAIN_EPOCHS=data["num_train_epochs"],
                                            PREPROCESSING_NUM_WORKERS=data["preprocessing_num_workers"],
                                            REVISION=data["revision"],
                                            RUNNING_MODE=data["running_mode"],
                                            SAVE_STRATEGY=data["save_strategy"],
                                            SEARCH_TYPE_METHOD=data["search_type_method"],
                                            TASK=data["task"],
                                            EARLY_STOPPING_PATIENCE=data["early_stopping_patience"]
                                            )

    file_name = './temporary_scripts/hyperparameter_search_' + data["task"] + "_" + str(
        data["gpu_number"]) + "_" + "_" + str(
        data["model_name"]).replace('/', '_') + "_" + time_now + ".sh"

    with open(file_name, "w") as f:
        print(final_command, file=f)

    return file_name


def run_script(command):
    try:
        os.system(command=command)
    except KeyboardInterrupt:
        print('Process interrupted.')


def run_in_parallel(commands_to_run):
    if len(commands_to_run) > 0:
        pool = Pool(processes=len(commands_to_run))
        pool.map(run_script, commands_to_run)


def process_gpu_number(gpu_number, do_hyperparameter_search):
    if gpu_number is None:
        gpu_number = [n for n in range(0, torch.cuda.device_count())]
        if len(gpu_number) == 0:
            if not do_hyperparameter_search:
                gpu_number = [None]  # In that case we use the CPU
            else:
                gpu_number = None
        else:
            if do_hyperparameter_search:
                gpu_number = ','.join(gpu_number)

    else:
        if not do_hyperparameter_search:
            if type(gpu_number) == str:
                if '_' in gpu_number:
                    gpu_number = [','.join(gpu_number.split('_'))]
                else:
                    gpu_number = gpu_number.split(',')
            elif type(gpu_number) == list:
                gpu_number = gpu_number
        else:
            if type(gpu_number) == list:
                gpu_number = ','.join(gpu_number)

    return gpu_number

def chunk_list(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def run_experiment(
        accumulation_steps,
        batch_size,
        dataset_cache_dir,
        do_hyperparameter_search,
        do_fp16,
        download_mode,
        eval_steps,
        evaluation_strategy,
        early_stopping_patience,
        gpu_memory,
        gpu_number,
        greater_is_better,
        hierarchical,
        language_model_type,
        learning_rate,
        list_of_languages,
        list_of_seeds,
        logging_steps,
        logging_strategy,
        lower_case,
        metric_for_best_model,
        preprocessing_num_workers,
        revision,
        running_mode,
        save_steps,
        save_strategy,
        search_type_method,
        task,
        log_directory=None,
        num_train_epochs=None,
        weight_decay=None,
        warmup_ratio=None
):
    # TODO I think it would be easier to just pass the whole data dictionary to the function
    #  so that we only have one parameter and do the same for the generate_command function

    preprocessing_num_workers = int(preprocessing_num_workers)

    batch_size_to_be_found = True if batch_size is None else False
    language_to_be_found = True if list_of_languages is None else False

    if list_of_languages is not None:
        if type(list_of_languages) == str:
            list_of_languages = list_of_languages.split(
                ',') if ',' in list_of_languages else [list_of_languages]

    if log_directory is None:
        log_directory = "results"
        time_stamp = re.sub(':', '_', datetime.datetime.now().isoformat())
        log_directory = log_directory + '/logs_' + str(time_stamp)
    else:
        if not os.path.isdir(log_directory):
            os.mkdir(log_directory)

    gpu_number = process_gpu_number(gpu_number, do_hyperparameter_search)

    # Tagging my Python scripts
    # setproctitle.setproctitle('Veton Matoshi - LEXTREME')
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpu_number])

    if type(list_of_seeds) == list:
        list_of_seeds = [int(s) for s in list_of_seeds if s]
    elif list_of_seeds is None:
        list_of_seeds = [1, 2, 3]  # run with 3 random seeds by default
    elif type(list_of_seeds) == str:
        list_of_seeds = list_of_seeds.split(',')
        list_of_seeds = [int(s) for s in list_of_seeds if s]

    print(list_of_seeds)

    # language_model_type is in the form {type: general|legal}_{language: ISO_CODE or "multilingual"]}_{size: small|base|large}
    language_model_info = language_model_type.split('_')
    if len(language_model_info) != 3:  # we got a direct language model just use it
        # This way we can comma-separate several language models
        models_to_be_used = language_model_type.split(',')
    else:  # find out what models we want to run
        types, languages, sizes = language_model_info[0], language_model_info[1], language_model_info[2]

        # expand comma-separated lists
        types = types.split(',') if ',' in types else [types]
        languages = languages.split(',') if ',' in languages else [languages]
        sizes = sizes.split(',') if ',' in sizes else [sizes]

        # expand 'all' keywords
        if 'all' in types:
            types = meta_infos["_TYPES"]
        if 'all' in languages:
            languages = meta_infos["_LANGUAGES"]
        if 'all' in sizes:
            sizes = meta_infos["_SIZES"]

        models_to_be_used = []
        for t in types:
            print(t)
            for l in languages:
                for s in sizes:
                    if l in meta_infos["language_models"][t]:
                        models_to_be_used.extend(
                            meta_infos["language_models"][t][l][s])  # add all models we want to run

    models_to_be_used = sorted(list(set(models_to_be_used)))
    print(models_to_be_used)

    gpu_command_dict = defaultdict(list)

    # Here we have to distinguish between running the experiments with our own hyperparameters and hyperparameter tuning

    if not do_hyperparameter_search:

        if task == 'all':
            all_variables = [[k for k, v in meta_infos["task_type_mapping"].items() if v != 'MCQA'], models_to_be_used,
                             list_of_seeds]
        elif isinstance(task, str) and ',' in task:
            tasks = task.split(',')
            all_variables = [[t.strip() for t in tasks], models_to_be_used,
                             list_of_seeds]
        else:
            all_variables = [[task], models_to_be_used, list_of_seeds]

        all_variables_perturbations = list(itertools.product(*all_variables))
        all_variables_perturbations = [
            '$'.join([str(x) for x in p]) for p in all_variables_perturbations]
        all_variables_perturbations = list(
            zip(cycle(gpu_number), all_variables_perturbations))
        all_variables_perturbations = [
            [x[0]] + x[1].split('$') for x in all_variables_perturbations]
        print(all_variables_perturbations)

        for (gpu_id, task, model_name, seed) in all_variables_perturbations:

            if bool(re.search(r'^\d+$', str(gpu_id))):
                gpu_id = int(gpu_id)
            seed = int(seed)

            if batch_size is None:
                batch_size, accumulation_steps = get_optimal_batch_size(
                    model_name, task, gpu_memory)
            else:
                if accumulation_steps is None:
                    accumulation_steps = 1

            if list_of_languages is None:
                if model_name in meta_infos["model_language_lookup_table"].keys():
                    # all means just the entire dataset, so that we take the default value
                    if meta_infos["model_language_lookup_table"][model_name] != 'all':
                        list_of_languages = [
                            meta_infos["model_language_lookup_table"][model_name]]
                    else:
                        list_of_languages = [None]

            for lang in list_of_languages:

                print("LANGUAGE IS: ", lang)

                epoch_and_strategies = get_default_number_of_training_epochs(task=task, model_name=model_name,
                                                                             language=lang, running_mode=running_mode)

                # eval_steps = epoch_and_strategies["eval_steps"]
                # logging_steps = epoch_and_strategies["logging_steps"]
                # save_steps = epoch_and_strategies["save_steps"]

                if num_train_epochs is None:
                    num_train_epochs = epoch_and_strategies["num_train_epochs"]
                if evaluation_strategy is None:
                    evaluation_strategy = epoch_and_strategies["evaluation_strategy"]
                if logging_strategy is None:
                    logging_strategy = epoch_and_strategies["logging_strategy"]
                if save_strategy is None:
                    save_strategy = epoch_and_strategies["save_strategy"]
                if logging_steps is None:
                    logging_steps = epoch_and_strategies["logging_steps"]
                if save_steps is None:
                    save_steps = epoch_and_strategies["save_steps"]
                if eval_steps is None:
                    eval_steps = epoch_and_strategies["eval_steps"]

                script_new = generate_command_for_experiments(
                    accumulation_steps=accumulation_steps,
                    batch_size=batch_size,
                    code=get_python_file_for_task(task),
                    dataset_cache_dir=dataset_cache_dir,
                    do_hyperparameter_search=do_hyperparameter_search,
                    do_fp16=do_fp16,
                    download_mode=download_mode,
                    eval_steps=eval_steps,
                    evaluation_strategy=evaluation_strategy,
                    early_stopping_patience=early_stopping_patience,
                    gpu_memory=gpu_memory,
                    gpu_number=gpu_id,
                    greater_is_better=greater_is_better,
                    hierarchical=hierarchical,
                    language=lang,
                    learning_rate=learning_rate,
                    log_directory=log_directory,
                    logging_steps=logging_steps,
                    logging_strategy=logging_strategy,
                    lower_case=lower_case,
                    metric_for_best_model=metric_for_best_model,
                    model_name=model_name,
                    num_train_epochs=num_train_epochs,
                    preprocessing_num_workers=preprocessing_num_workers,
                    revision=revision,
                    running_mode=running_mode,
                    save_steps=save_steps,
                    save_strategy=save_strategy,
                    seed=seed,
                    task=task,
                    warmup_ratio=warmup_ratio,
                    weight_decay=weight_decay
                )

                if script_new is not None:
                    # sleep different time for each seed to prevent access to temporary scripts at the same time
                    command = f'sleep {seed * 10}; bash {str(script_new)}'
                    gpu_command_dict[gpu_id].append(command)
                    print(command)

            if batch_size_to_be_found:
                # Have to set batch_size back to None, otherwise it will continue to assign too high batch sizes which will cause errors
                batch_size = None

            if language_to_be_found:
                list_of_languages = None  # Set language back to None

    else:
        if task == 'all':
            all_variables = [
                [k for k, v in meta_infos["task_type_mapping"].items() if v != 'MCQA'], models_to_be_used]
        else:
            all_variables = [[task], models_to_be_used]

        all_variables_perturbations = list(itertools.product(*all_variables))
        all_variables_perturbations = [
            '$'.join([str(x) for x in p]) for p in all_variables_perturbations]
        all_variables_perturbations = list(
            zip(cycle([gpu_number]), all_variables_perturbations))
        all_variables_perturbations = [
            [x[0]] + x[1].split('$') for x in all_variables_perturbations]
        print(all_variables_perturbations)

        for (gpu_id, task, model_name) in all_variables_perturbations:

            if list_of_languages is None:
                if model_name in meta_infos["model_language_lookup_table"].keys():
                    # all means just the entire dataset, so that we take the default value
                    if meta_infos["model_language_lookup_table"][model_name] != 'all':
                        list_of_languages = [
                            meta_infos["model_language_lookup_table"][model_name]]
                    else:
                        list_of_languages = [None]

            for lang in list_of_languages:

                print("LANGUAGE IS: ", lang)

                epoch_and_strategies = get_default_number_of_training_epochs(task=task, model_name=model_name,
                                                                             language=lang, running_mode=running_mode)

                # eval_steps = epoch_and_strategies["eval_steps"]
                # logging_steps = epoch_and_strategies["logging_steps"]
                # save_steps = epoch_and_strategies["save_steps"]

                if num_train_epochs is None:
                    num_train_epochs = epoch_and_strategies["num_train_epochs"]
                if evaluation_strategy is None:
                    evaluation_strategy = epoch_and_strategies["evaluation_strategy"]
                if logging_strategy is None:
                    logging_strategy = epoch_and_strategies["logging_strategy"]
                if save_strategy is None:
                    save_strategy = epoch_and_strategies["save_strategy"]
                if logging_steps is None:
                    logging_steps = epoch_and_strategies["logging_steps"]
                if save_steps is None:
                    save_steps = epoch_and_strategies["save_steps"]
                if eval_steps is None:
                    eval_steps = epoch_and_strategies["eval_steps"]

                script_new = generate_command_for_hyperparameter_search(
                    accumulation_steps=accumulation_steps,
                    batch_size=batch_size,
                    code=get_python_file_for_task(task),
                    dataset_cache_dir=dataset_cache_dir,
                    do_hyperparameter_search=do_hyperparameter_search,
                    download_mode=download_mode,
                    eval_steps=eval_steps,
                    evaluation_strategy=evaluation_strategy,
                    early_stopping_patience=early_stopping_patience,
                    gpu_memory=gpu_memory,
                    gpu_number=gpu_id,
                    greater_is_better=greater_is_better,
                    hierarchical=hierarchical,
                    language=lang,
                    learning_rate=learning_rate,
                    log_directory=log_directory,
                    logging_steps=logging_steps,
                    logging_strategy=logging_strategy,
                    lower_case=lower_case,
                    metric_for_best_model=metric_for_best_model,
                    model_name=model_name,
                    num_train_epochs=num_train_epochs,
                    preprocessing_num_workers=preprocessing_num_workers,
                    revision=revision,
                    running_mode=running_mode,
                    save_steps=save_steps,
                    save_strategy=save_strategy,
                    search_type_method=search_type_method,
                    task=task,
                    warmup_ratio=warmup_ratio,
                    weight_decay=weight_decay
                )

                if script_new is not None:
                    # sleep different time for each seed to prevent access to temporary scripts at the same time
                    command = f'sleep {random.choice([1, 2, 3, 4]) * 10}; bash {str(script_new)}'
                    gpu_command_dict[gpu_id].append(command)
                    print(command)

            if language_to_be_found:
                list_of_languages = None  # Set language back to None

    with open('command_dict.json', 'w') as f:
        js.dump(gpu_command_dict, f, ensure_ascii=False, indent=2)



    with open('temporary_scripts/final_commands_run_inparallel.txt', 'w') as f:
        for gpu_id, commands in gpu_command_dict.items():
            print('On GPU ', gpu_id,
                  'this system command is applied:', file=f, end='\n')
            print('\t', '; sleep 10; '.join(commands), file=f, end='\n')
            print('On GPU ', gpu_id,
                  'the following commands are run sequentially.', file=f, end='\n')
            for c in commands:
                print('\t', c, file=f)
            print("\n########################################\n", file=f)

    # commands_to_run = ['; sleep 10; '.join(commands) for _, commands in gpu_command_dict.items()]
    commands_to_run = [commands for commands in gpu_command_dict.values()]
    commands_to_run_zipped = list(itertools.zip_longest(*commands_to_run))

    # commands_to_run_zipped_chunked = chunk_list(commands_to_run_zipped, len(gpu_command_dict.keys()))
    for cmds in commands_to_run_zipped:
        cmds = list(cmds)
        cmds = [c for c in cmds if c is not None]
        run_in_parallel(commands_to_run=cmds)
        sleep(3)

