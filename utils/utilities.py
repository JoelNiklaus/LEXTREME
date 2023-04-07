import arrow
from collections import defaultdict
import json as js
import os
from pathlib import Path
import datetime
import re


def remove_old_files(path_to_directory, days=-30):
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
        'xlm-roberta-base': {256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2},
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0},  # model is too large
        'xlm-roberta-large': {256: 0, 512: 0, 1024: 0, 2048: 0, 4096: 0},  # model is too large
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
        print(f"Did not find a batch size for the language model {language_model} and the gpu memory {gpu_memory}")
        if "joelito" in language_model:
            if "base" in language_model:
                print(f"We assume that the language model {language_model} is based on xlm-roberta-base.")
                batch_size_dict = optimal_batch_sizes[int(gpu_memory)]["xlm-roberta-base"]
            elif "large" in language_model:
                print(f"We assume that the language model {language_model} is based on xlm-roberta-large.")
                batch_size_dict = optimal_batch_sizes[int(gpu_memory)]["xlm-roberta-large"]
            else:
                raise ValueError("Did not find the language model in the dictionary.")
        else:
            print(f"The language model {language_model} will be considered a monolingual model. "
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


def generate_command(**data):
    time_now = datetime.datetime.now().isoformat()

    if data["hierarchical"] is None:
        data["hierarchical"] = get_hierarchical(data["task"])

    command_template_general = 'python3 ./experiments/{CODE} ' \
                               '--model_name_or_path {MODEL_NAME} ' \
                               '--log_directory {LOG_DIRECTORY} ' \
                               '--do_train ' \
                               '--do_eval ' \
                               '--do_predict ' \
                               '--overwrite_output_dir ' \
                               '--load_best_model_at_end ' \
                               '--metric_for_best_model {METRIC_FOR_BEST_MODEL} ' \
                               '--greater_is_better {GREATER_IS_BETTER} ' \
                               '--evaluation_strategy {EVALUATION_STRATEGY} ' \
                               '--logging_strategy {LOGGING_STRATEGY} ' \
                               '--save_strategy {SAVE_STRATEGY} ' \
                               '--save_total_limit 6 ' \
                               '--num_train_epochs {NUM_TRAIN_EPOCHS} ' \
                               '--learning_rate {LEARNING_RATE} ' \
                               '--per_device_train_batch_size {BATCH_SIZE} ' \
                               '--per_device_eval_batch_size {BATCH_SIZE} ' \
                               '--seed {SEED} ' \
                               '--gradient_accumulation_steps {ACCUMULATION_STEPS} ' \
                               '--eval_accumulation_steps {ACCUMULATION_STEPS} ' \
                               '--running_mode {RUNNING_MODE} ' \
                               '--download_mode {DOWNLOAD_MODE} ' \
                               '--preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} ' \
                               '--hierarchical {HIERARCHICAL} ' \
                               '--revision {REVISION} ' \
                               '--affair_text_scope {AFFAIR_TEXT_SCOPE} ' \


    command_template_for_hyperparameter_search = 'python3 ./experiments/{CODE} ' \
                                                 '--model_name_or_path {MODEL_NAME} ' \
                                                 '--log_directory {LOG_DIRECTORY} ' \
                                                 '--running_mode {RUNNING_MODE} ' \
                                                 '--download_mode {DOWNLOAD_MODE} ' \
                                                 '--preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} ' \
                                                 '--hierarchical {HIERARCHICAL} ' \
                                                 '--revision {REVISION} ' \
                                                 '--affair_text_scope {AFFAIR_TEXT_SCOPE} '

    if "do_hyperparameter_search" in data.keys() and data["do_hyperparameter_search"]:
        command_template = command_template_for_hyperparameter_search
        command_template += ' --do_hyperparameter_search'
    else:
        command_template = command_template_general

    if data["dataset_cache_dir"] is not None:
        command_template = command_template + ' --dataset_cache_dir {DATASET_CACHE_DIR}'

    if data["language"] is not None:
        command_template = command_template + ' --language {LANGUAGE} '
        command_template = command_template + '--output_dir {LOG_DIRECTORY}/{TASK}/{' \
                                              'MODEL_NAME}/results_after_training_only_on_language_with_id__{' \
                                              'LANGUAGE}/seed_{SEED}'
    else:
        command_template = command_template + ' --output_dir {LOG_DIRECTORY}/{TASK}/{MODEL_NAME}/seed_{SEED} '

    command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} ' + command_template
    run_on_cpu = "gpu_number" not in data.keys() or not bool(re.search("\d", str(data["gpu_number"])))
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

    if "do_hyperparameter_search" in data.keys():
        if not data["do_hyperparameter_search"]:  # For hyperparameter search we do not need this
            if 'logging_steps' in data.keys() and data['logging_steps'] is not None:
                command_template += ' --logging_steps ' + str(data["logging_steps"]) + ' '
            if 'eval_steps' in data.keys() and data['eval_steps'] is not None:
                command_template += ' --eval_steps ' + str(data["eval_steps"]) + ' '
            if 'save_steps' in data.keys() and data['save_steps'] is not None:
                command_template += ' --save_steps ' + str(data["save_steps"]) + ' '

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
                                            AFFAIR_TEXT_SCOPE=data["affair_text_scope"]
                                            )

    file_name = './temporary_scripts/' + data["task"] + "_" + str(data["gpu_number"]) + "_" + str(
        data["seed"]) + "_" + str(data["model_name"]).replace('/', '_') + "_" + time_now + ".sh"

    with open(file_name, "w") as f:
        print(final_command, file=f)

    return file_name

# Get the evaluation_strategy, logging_strategy, save_strategy, eval_steps, logging_steps
