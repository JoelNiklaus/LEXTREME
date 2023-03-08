import arrow
from collections import defaultdict
import json as js
import os
from pathlib import Path


def remove_old_files(path_to_directory, days=-7):
    # keep files for 7 days to make sure that this works also in a hpc environment
    criticalTime = arrow.now().shift().shift(days=days)
    print(f"Removing old temporary bash scripts (older than {days} days: before {criticalTime})")

    for file in Path(path_to_directory).glob('*'):
        if file.is_file():
            file_time_modified = arrow.get(file.stat().st_mtime)
            if file_time_modified < criticalTime:
                try:
                    os.remove(file)
                except OSError:
                    pass  # we don't care if the file is not there


def get_meta_infos():
    path_to_json_file = os.path.join(os.path.dirname(__file__), 'meta_infos.json')
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

    return meta_infos


meta_infos = get_meta_infos()

optimal_batch_sizes = {
    # e.g. RTX 2080 Ti or GTX 1080 Ti
    # TODO test sizes here
    11: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 4},  # untested
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        'monolingual': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2},
        'xlm-roberta-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2},  # untested
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2},  # untested
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 1},  # untested
    },
    # TODO test sizes here
    # e.g. P100
    16: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16},  # untested
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8},  # untested
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        'monolingual': {256: 32, 512: 32, 1024: 8, 2048: 4, 4096: 4},
        'xlm-roberta-base': {256: 32, 512: 32, 1024: 16, 2048: 4, 4096: 4},  # untested
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2},  # untested
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2},  # untested
    },
    # e.g. RTX 3090
    24: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8},
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        'monolingual': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8},
        'xlm-roberta-base': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8},
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2},
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2},
    },
    # e.g. V100
    32: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 64, 2048: 32, 4096: 16},
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        'monolingual': {256: 64, 512: 32, 1024: 32, 2048: 8, 4096: 8},
        'xlm-roberta-base': {256: 32, 512: 16, 1024: 16, 2048: 8, 4096: 4},
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 8, 4096: 8},
        'xlm-roberta-large': {256: 8, 512: 4, 1024: 4, 2048: 2, 4096: 2},
    },
    # TODO test sizes here
    # e.g. A6000
    48: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 64},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        'monolingual': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},
        'xlm-roberta-base': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 16},
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16},  # bf16
        # 'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # fp32
        'xlm-roberta-large': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # fp16
    },
    # e.g. A100
    80: {
        'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 64},  # fp16
        # 'distilbert-base-multilingual-cased': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},  # fp32
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},  # fp16
        # 'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},  # fp32
        # same as xlm-r to be safe (monolingual models have a smaller vocab than xlm-r and are equally sized
        'monolingual': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 32},
        'xlm-roberta-base': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 16},  # fp16
        # 'xlm-roberta-base': {256: 64, 512: 64, 1024: 64, 2048: 64, 4096: 16},  # fp32
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 64, 2048: 32, 4096: 16},  # bf16
        # 'microsoft/mdeberta-v3-base': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # fp32
        'xlm-roberta-large': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 8},  # fp16
        # 'xlm-roberta-large': {256: 64, 512: 64, 1024: 32, 2048: 16, 4096: 4}, # fp32
    },
}

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
        print("The language model ", language_model, " will be considered a monolingual model. "
                                                     "Therefore, we revert to the default batch size.")
        batch_size_dict = optimal_batch_sizes[int(gpu_memory)]["monolingual"]
    batch_size = None
    while batch_size is None:
        try:
            batch_size = batch_size_dict[max_seq_len]
        except KeyError:
            max_seq_len *= 2  # if the sequence length is not in the dictionary, we try the next higher one
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

        else:
            num_train_epochs = 50
            num_train_epochs = 50
            evaluation_strategy = "epoch"
            logging_strategy = "epoch"
            save_strategy = "epoch"

    else:
        num_train_epochs = 50
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

# Get the evaluation_strategy, logging_strategy, save_strategy, eval_steps, logging_steps
