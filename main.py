import argparse
import datetime
import itertools
import json as js
import os
import re
import shutil
from collections import defaultdict
from itertools import cycle
from multiprocessing import Pool

import setproctitle
import torch

# Empty folder with temporary scripts
# TODO remove temporary scripts that are older than one day to clean up (keep most recent ones so that
# shutil.rmtree('./temporary_scripts', ignore_errors=True) # Uncomment if you want to delete all temporary scripts
os.makedirs('./temporary_scripts', exist_ok=True)

multilingual_models = {
    "small": [
        "distilbert-base-multilingual-cased",
        "microsoft/Multilingual-MiniLM-L12-H384"
    ],
    "base": [
        "microsoft/mdeberta-v3-base",  # TODO test batch sizes for this model again because we removed fp16
        # "xlm-roberta-base"
    ],
    "large": [
        "xlm-roberta-large"
    ]
}

optimal_batch_sizes = {
    # e.g. RTX 3090
    24: {
        'distilbert-base-multilingual-cased': {512: 64, 1024: 64, 2048: 32, 4096: 16},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8},
        'xlm-roberta-base': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8},
        'microsoft/mdeberta-v3-base': {256: 64, 512: 16, 1024: 16, 2048: 8, 4096: 4},
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2},
    },
    # e.g. V100
    # TODO test if you can fit larger batch sizes for sequence lengths in [256, 1024, 2048]
    32: {
        'distilbert-base-multilingual-cased': {512: 64, 1024: 64, 2048: 32, 4096: 16},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8},
        'xlm-roberta-base': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8},
        'microsoft/mdeberta-v3-base': {256: 64, 512: 16, 1024: 16, 2048: 8, 4096: 4},
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2},
    },
    # e.g. A100
    # TODO test if we can fit larger batch sizes
    80: {
        'distilbert-base-multilingual-cased': {512: 64, 1024: 64, 2048: 32, 4096: 16},
        'microsoft/Multilingual-MiniLM-L12-H384': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8},
        'xlm-roberta-base': {256: 64, 512: 32, 1024: 16, 2048: 8, 4096: 8},
        'microsoft/mdeberta-v3-base': {256: 64, 512: 16, 1024: 16, 2048: 8, 4096: 4},
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 8, 2048: 4, 4096: 2},
    },
    # TODO test this for different sequence lengths
}

# TODO get this directly from the LEXTREME huggingface dataset loader
# SLTC: Single Class Text Classification
# MLTC: Multi Class Text Classification
# NER: Named Entity Recognition
task_code_mapping = {
    'brazilian_court_decisions_judgment': 'SLTC',
    'brazilian_court_decisions_unanimity': 'SLTC',
    'german_argument_mining': 'SLTC',
    'greek_legal_code_chapter_level': 'SLTC',
    'greek_legal_code_subject_level': 'SLTC',
    'greek_legal_code_volume_level': 'SLTC',
    'swiss_judgment_prediction': 'SLTC',
    'online_terms_of_service_unfairness_level': 'SLTC',
    'online_terms_of_service_unfairness_category': 'MLTC',
    'covid19_emergency_event': 'MLTC',
    'multi_eurlex_level_1': 'MLTC',
    'multi_eurlex_level_2': 'MLTC',
    'multi_eurlex_level_3': 'MLTC',
    'greek_legal_ner': 'NER',
    'legalnero': 'NER',
    'lener_br': 'NER',
    'mapa_ner_coarse_grained': 'NER',
    'mapa_ner_fine_grained': 'NER',
}

max_sequence_lengths = {  # 256, 512, 1024, 2048, 4096
    'brazilian_court_decisions_judgment': 8 * 128,  # 1024
    'brazilian_court_decisions_unanimity': 8 * 128,  # 1024
    'covid19_emergency_event': 256,
    'german_argument_mining': 256,
    'greek_legal_code_chapter_level': 32 * 128,  # 4096
    'greek_legal_code_subject_level': 32 * 128,  # 4096
    'greek_legal_code_volume_level': 32 * 128,  # 4096
    'greek_legal_ner': 512,
    'legalnero': 512,
    'lener_br': 512,
    'mapa_ner_fine_grained': 512,
    'mapa_ner_coarse_grained': 512,
    'multi_eurlex_level_1': 32 * 128,  # 4096
    'multi_eurlex_level_2': 32 * 128,  # 4096
    'multi_eurlex_level_3': 32 * 128,  # 4096
    'online_terms_of_service_unfairness_category': 256,
    'online_terms_of_service_unfairness_level': 256,
    'swiss_judgment_prediction': 16 * 128,  # 2048
}


def get_python_file_for_task(task):
    return f"run_{task}.py"


def generate_command(time_now, **data):
    command_template = 'python3 ./experiments/{CODE} ' \
                       '--model_name_or_path {MODEL_NAME} ' \
                       '--output_dir {OUTPUT_DIR}/{TASK}/{MODEL_NAME}/seed_{SEED} ' \
                       '--do_train --do_eval --do_predict ' \
                       '--overwrite_output_dir ' \
                       '--load_best_model_at_end --metric_for_best_model {METRIC_FOR_BEST_MODEL} ' \
                       '--greater_is_better {GREATER_IS_BETTER} ' \
                       '--evaluation_strategy epoch --save_strategy epoch ' \
                       '--save_total_limit 5 ' \
                       '--num_train_epochs {NUM_TRAIN_EPOCHS} ' \
                       '--learning_rate {LEARNING_RATE} ' \
                       '--per_device_train_batch_size {BATCH_SIZE} --per_device_eval_batch_size {BATCH_SIZE} ' \
                       '--seed {SEED} ' \
                       '--gradient_accumulation_steps {ACCUMULATION_STEPS} --eval_accumulation_steps {ACCUMULATION_STEPS} ' \
                       '--running_mode {RUNNING_MODE} ' \
                       '--download_mode {DOWNLOAD_MODE} ' \
                       '--preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} '

    if data["dataset_cache_dir"] is not None:
        command_template = command_template + '--dataset_cache_dir {DATASET_CACHE_DIR}'

    if "gpu_number" not in data.keys() or not bool(re.search("\d", str(data["gpu_number"]))):
        data["gpu_number"] = ""
        # If no GPU available, we cannot make use of --fp16 --fp16_full_eval
        command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} ' + command_template
    elif data["model_name"] == "microsoft/mdeberta-v3-base":
        # mdeberta does not work with fp16 because it was trained with bf16
        # probably similar for MobileBERT: https://github.com/huggingface/transformers/issues/11327
        # For some reason microsoft/mdeberta-v3-base token classification returns eval_loss == NaN when using fp16
        command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} ' + command_template
    else:
        # --fp16_full_eval removed because they cause errors: transformers RuntimeError: expected scalar type Half but found Float
        command_template = 'CUDA_VISIBLE_DEVICES={GPU_NUMBER} ' + command_template + ' --fp16'

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
                                            OUTPUT_DIR=data["output_dir"],
                                            PREPROCESSING_NUM_WORKERS=data["preprocessing_num_workers"],
                                            DATASET_CACHE_DIR=data["dataset_cache_dir"]
                                            )

    if "hierarchical" in data.keys() and data["hierarchical"] is not None:
        final_command += ' --hierarchical ' + data["hierarchical"]

    file_name = './temporary_scripts/' + data["task"] + "_" + str(data["gpu_number"]) + "_" + str(
        data["seed"]) + "_" + str(data["model_name"]).replace('/', '_') + "_" + time_now + ".sh"

    with open(file_name, "w") as f:
        print(final_command, file=f)

    return file_name


def get_optimal_batch_size(language_model: str, task: str, gpu_memory, total_batch_size=64):
    max_seq_len = max_sequence_lengths[task]

    batch_size_dict = optimal_batch_sizes[int(gpu_memory)][language_model]
    batch_size = None
    while batch_size is None:
        try:
            batch_size = batch_size_dict[max_seq_len]
        except KeyError:
            max_seq_len *= 2  # if the sequence length is not in the dictionary, we try the next higher one
    accumulation_steps = total_batch_size // batch_size

    return batch_size, accumulation_steps


def run_script(command):
    try:
        os.system(command=command)
    except KeyboardInterrupt:
        print('Process interrupted.')


def run_in_parallel(commands_to_run):
    if len(commands_to_run) > 0:
        pool = Pool(processes=len(commands_to_run))
        pool.map(run_script, commands_to_run)


def run_experiment(running_mode, download_mode, language_model_type, task, list_of_seeds, batch_size,
                   accumulation_steps, lower_case, language, learning_rate, gpu_number, gpu_memory, hierarchical,
                   preprocessing_num_workers,
                   dataset_cache_dir, num_train_epochs=None, output_dir=None):
    time_stamp = datetime.datetime.now().isoformat()

    preprocessing_num_workers = int(preprocessing_num_workers)

    if batch_size is None:
        batch_size_to_be_found = True
    else:
        batch_size_to_be_found = False

    if num_train_epochs is None:
        if 'multi_eurlex' in task:
            num_train_epochs = 1  # this dataset is so large, one epoch is enough to save compute
        else:
            num_train_epochs = 50
    if output_dir is None:
        output_dir = 'results/logs_' + str(time_stamp)

    if gpu_number is None:
        gpu_number = [n for n in range(0, torch.cuda.device_count())]
        if len(gpu_number) == 0:
            gpu_number = [None]  # In that case we use the CPU
    else:
        if type(gpu_number) == str:
            gpu_number = gpu_number.split(',')
        elif type(gpu_number) == list:
            gpu_number = gpu_number

    # Tagging my Python scripts
    setproctitle.setproctitle('Veton Matoshi - LEXTREME')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpu_number])

    if type(list_of_seeds) == list:
        list_of_seeds = [int(s) for s in list_of_seeds if s]
    elif list_of_seeds is None:
        list_of_seeds = [1, 2, 3]  # run with 3 random seeds by default
    elif type(list_of_seeds) == str:
        list_of_seeds = list_of_seeds.split(',')
        list_of_seeds = [int(s) for s in list_of_seeds if s]

    print(list_of_seeds)

    # TODO add monolingual models
    if language_model_type == 'all':
        models_to_be_used = multilingual_models['small'] + multilingual_models['base'] + multilingual_models['large']
    else:
        models_to_be_used = multilingual_models[language_model_type]

    models_to_be_used = sorted(list(set(models_to_be_used)))
    print(models_to_be_used)

    gpu_command_dict = defaultdict(list)

    metric_for_best_model = "eval_loss"
    greater_is_better = False

    if task == 'all':
        all_variables = [[t for t in list(task_code_mapping.keys())], models_to_be_used, list_of_seeds]
    else:
        all_variables = [[task], models_to_be_used, list_of_seeds]

    all_variables_perturbations = list(itertools.product(*all_variables))
    all_variables_perturbations = ['$'.join([str(x) for x in p]) for p in all_variables_perturbations]
    all_variables_perturbations = list(zip(cycle(gpu_number), all_variables_perturbations))
    all_variables_perturbations = [[x[0]] + x[1].split('$') for x in all_variables_perturbations]
    print(all_variables_perturbations)

    for (gpu_id, task, model_name, seed) in all_variables_perturbations:
        if bool(re.search('\d', str(gpu_id))):
            gpu_id = int(gpu_id)
        seed = int(seed)
        if batch_size is None:
            batch_size, accumulation_steps = get_optimal_batch_size(model_name, task, gpu_memory)
        else:
            if accumulation_steps is None:
                accumulation_steps = 1
        script_new = generate_command(time_now=time_stamp, gpu_number=gpu_id, model_name=model_name,
                                      lower_case=lower_case, task=task, seed=seed,
                                      num_train_epochs=num_train_epochs, batch_size=batch_size,
                                      accumulation_steps=accumulation_steps, language=language,
                                      running_mode=running_mode, learning_rate=learning_rate,
                                      code=get_python_file_for_task(task), metric_for_best_model=metric_for_best_model,
                                      hierarchical=hierarchical, greater_is_better=greater_is_better,
                                      download_mode=download_mode, output_dir=output_dir,
                                      preprocessing_num_workers=preprocessing_num_workers,
                                      dataset_cache_dir=dataset_cache_dir)
        if batch_size_to_be_found:
            # Have to set batch_size back to None, otherwise it will continue to assign too high batch sizes which will cause errors
            batch_size = None
        if script_new is not None:
            # sleep different time for each seed to prevent access to temporary scripts at the same time
            command = f'sleep {seed * 10}; bash {str(script_new)}'
            gpu_command_dict[gpu_id].append(command)
            print(command)

    with open('command_dict.json', 'w') as f:
        js.dump(gpu_command_dict, f, ensure_ascii=False, indent=2)

    commands_to_run = ['; sleep 10; '.join(commands) for _, commands in gpu_command_dict.items()]

    with open('temporary_scripts/final_commands_run_inparallel.txt', 'w') as f:
        for gpu_id, commands in gpu_command_dict.items():
            print('On GPU ', gpu_id, 'this system command is applied:', file=f, end='\n')
            print('\t', '; sleep 10; '.join(commands), file=f, end='\n')
            print('On GPU ', gpu_id, 'the following commands are run sequentially.', file=f, end='\n')
            for c in commands:
                print('\t', c, file=f)
            print("\n########################################\n", file=f)

    run_in_parallel(commands_to_run=commands_to_run)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-as', '--accumulation_steps', help='Define the number of accumulation_steps.', default=None)
    parser.add_argument('-bz', '--batch_size', help='Define the batch size.', default=None)
    parser.add_argument('-gn', '--gpu_number', help='Define which GPU you would like to use.', default=None)
    parser.add_argument('-gm', '--gpu_memory', help='Define how much memory your GPUs have', default=None)
    parser.add_argument('-hier', '--hierarchical',
                        help='Define whether you want to use a hierarchical model or not. '
                             'Caution: this will not work for every task',
                        default=None)
    parser.add_argument('-lang', '--language', help='Define if you want to filter the training dataset by language.',
                        default='all_languages')
    parser.add_argument('-lc', '--lower_case', help='Define if lower case or not.', default=False)
    parser.add_argument('-lmt', '--language_model_type',
                        help='Define which kind of language model you would like to use; '
                             'you can choose between small, base and large language models or all of them.',
                        default='all')
    parser.add_argument('-los', '--list_of_seeds',
                        help='Define the random seeds for which you want to run the experiments.',
                        default=None)
    parser.add_argument('-lr', '--learning_rate', help='Define the learning rate', default=1e-5)
    parser.add_argument('-nte', '--num_train_epochs', help='Define the number of training epochs.')
    parser.add_argument('-rmo', '--running_mode',
                        help='Define whether you want to run the finetuning on all available training data ("default"), '
                             'on 100 examples for debugging ("debug") '
                             'or on 5% for testing purposes ("experimental").',
                        default='default')
    parser.add_argument('-dmo', '--download_mode',
                        help='Define whether you want to redownload the dataset or not. '
                             'See the options in: https://huggingface.co/docs/datasets/v1.5.0/loading_datasets.html#download-mode',
                        default='reuse_cache_if_exists')  # reuses raw downloaded files but makes dataset freshly
    parser.add_argument('-t', '--task', help='Choose a task.', default='all',
                        choices=sorted(list(task_code_mapping.keys())))
    parser.add_argument('-od', '--output_dir', help='Choose an output directory.', default=None)
    parser.add_argument('-nw', '--preprocessing_num_workers',
                        help="The number of processes to use for the preprocessing. "
                             "If it deadlocks, try setting this to 1.", default=1)
    parser.add_argument('-cad', '--dataset_cache_dir',
                        help="Specify the directory you want to cache your datasets.",
                        default=None)

    args = parser.parse_args()

    # TODO replace this with variable from argparse
    if args.download_mode == 'force_redownload':
        # Remove the existing cache directory since everything will be redownloaded anyway
        # Somehow the cache caused errors
        if os.path.isdir('datasets_caching'):
            shutil.rmtree('datasets_caching')
        else:
            os.mkdir('datasets_caching')

    run_experiment(
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        dataset_cache_dir=args.dataset_cache_dir,
        download_mode=args.download_mode,
        gpu_memory=args.gpu_memory,
        gpu_number=args.gpu_number,
        hierarchical=args.hierarchical,
        language=args.language,
        language_model_type=args.language_model_type,
        learning_rate=args.learning_rate,
        list_of_seeds=args.list_of_seeds,
        lower_case=args.lower_case,
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        preprocessing_num_workers=args.preprocessing_num_workers,
        running_mode=args.running_mode,
        task=args.task
    )
