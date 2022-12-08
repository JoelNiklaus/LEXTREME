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
from utils.utilities import remove_old_files

import setproctitle
import torch

# Empty folder with temporary scripts
# shutil.rmtree('./temporary_scripts', ignore_errors=True) # Uncomment if you want to delete all temporary scripts
os.makedirs('./temporary_scripts', exist_ok=True)
remove_old_files('./temporary_scripts')

_TYPES = ['general', 'legal']
_LANGUAGES = ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga', 'hr',
              'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']
_SIZES = ['small', 'base', 'large']

# TODO look if there are other models available that we want to run: e.g. small or large ones
language_models = {
    "general": {
        "bg": {
            "small": [],
            "base": ["iarfmoose/roberta-base-bulgarian"],
            "large": []
        },
        "cs": {
            "small": [],
            "base": [],
            "large": []
        },
        "da": {
            "small": [],
            "base": ["Maltehb/danish-bert-botxo"],
            "large": []
        },
        "de": {
            "small": [],
            "base": ["dbmdz/bert-base-german-cased", "deepset/gbert-base"],
            "large": []
        },
        "el": {
            "small": [],
            "base": ["nlpaueb/bert-base-greek-uncased-v1"],
            "large": []
        },
        "en": {
            "small": [],
            "base": ["roberta-base"],  # TODO maybe test more etc.
            "large": []
        },
        "es": {
            "small": [],
            "base": ["bertin-project/bertin-roberta-base-spanish", "PlanTL-GOB-ES/roberta-base-bne"],
            "large": []
        },
        "et": {
            "small": [],
            "base": [],
            "large": []
        },
        "fi": {
            "small": [],
            "base": ["TurkuNLP/bert-base-finnish-cased-v1"],
            "large": []
        },
        "fr": {
            "small": [],
            "base": ["camembert-base", "dbmdz/bert-base-french-europeana-cased"],
            "large": []
        },
        "ga": {
            "small": [],
            "base": ["DCU-NLP/bert-base-irish-cased-v1"],
            "large": []
        },
        "hr": {
            "small": [],
            "base": [],
            "large": []
        },
        "hu": {
            "small": [],
            "base": [],
            "large": []
        },
        "it": {
            "small": [],
            "base": ["Musixmatch/umberto-commoncrawl-cased-v1", "dbmdz/bert-base-italian-cased"],
            "large": []
        },
        "lt": {
            "small": [],
            "base": [],
            "large": []
        },
        "lv": {
            "small": [],
            "base": [],
            "large": []
        },
        "mt": {
            "small": [],
            "base": [],
            "large": []
        },
        "nl": {
            "small": [],
            "base": ["GroNLP/bert-base-dutch-cased"],
            "large": []
        },
        "pl": {
            "small": [],
            "base": ["dkleczek/bert-base-polish-uncased-v1"],
            "large": []
        },
        "pt": {
            "small": [],
            "base": ["neuralmind/bert-base-portuguese-cased"],
            "large": []
        },
        "ro": {
            "small": [],
            "base": ["dumitrescustefan/bert-base-romanian-uncased-v1"],
            "large": []
        },
        "sk": {
            "small": [],
            "base": ["gerulata/slovakbert"],
            "large": []
        },
        "sl": {
            "small": [],
            "base": [],
            "large": []
        },
        "sv": {
            "small": [],
            "base": ["KB/bert-base-swedish-cased"],
            "large": []
        },
        "multilingual": {
            "small": [
                "distilbert-base-multilingual-cased",
                "microsoft/Multilingual-MiniLM-L12-H384",
            ],
            "base": [
                "xlm-roberta-base",
                "microsoft/mdeberta-v3-base",
            ],
            "large": [
                "xlm-roberta-large",
            ]
        },
    },
    "legal": {
        "it": {
            "small": [],
            "base": ["dlicari/Italian-Legal-BERT"],
            "large": []
        },
        "ro": {
            "small": [],
            "base": ["readerbench/jurBERT-base"],
            "large": []
        },
        "en": {
            "small": [],
            "base": ["zlucia/custom-legalbert", "nlpaueb/legal-bert-base-uncased"],
            "large": []
        },
        "es": {
            "small": [],
            "base": ["PlanTL-GOB-ES/RoBERTalex"],
            "large": []
        },
    }  # TODO add more legal models maybe from nllpw: https://nllpw.org/workshop/program/
}

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

# TODO move these completely to meta_infos.json
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
        'xlm-roberta-base': {256: 64, 512: 32, 1024: 32, 2048: 16, 4096: 8},
        # lower batch sizes because not possible with fp16
        'microsoft/mdeberta-v3-base': {256: 32, 512: 16, 1024: 8, 2048: 8, 4096: 8},
        'xlm-roberta-large': {256: 16, 512: 8, 1024: 16, 2048: 8, 4096: 8},
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
task_code_mapping = {
    'brazilian_court_decisions_judgment': 'SLTC',
    'brazilian_court_decisions_unanimity': 'SLTC',
    'german_argument_mining': 'SLTC',
    'greek_legal_code_chapter': 'SLTC',
    'greek_legal_code_subject': 'SLTC',
    'greek_legal_code_volume': 'SLTC',
    'swiss_judgment_prediction': 'SLTC',
    'online_terms_of_service_unfairness_levels': 'SLTC',
    'online_terms_of_service_clause_topics': 'MLTC',
    'covid19_emergency_event': 'MLTC',
    'multi_eurlex_level_1': 'MLTC',
    'multi_eurlex_level_2': 'MLTC',
    'multi_eurlex_level_3': 'MLTC',
    'greek_legal_ner': 'NER',
    'legalnero': 'NER',
    'lener_br': 'NER',
    'mapa_coarse': 'NER',
    'mapa_fine': 'NER',
}

max_sequence_lengths = {  # 256, 512, 1024, 2048, 4096
    'brazilian_court_decisions_judgment': 8 * 128,  # 1024
    'brazilian_court_decisions_unanimity': 8 * 128,  # 1024
    'covid19_emergency_event': 256,
    'german_argument_mining': 256,
    'greek_legal_code_chapter': 32 * 128,  # 4096
    'greek_legal_code_subject': 32 * 128,  # 4096
    'greek_legal_code_volume': 32 * 128,  # 4096
    'greek_legal_ner': 512,
    'legalnero': 512,
    'lener_br': 512,
    'mapa_fine': 512,
    'mapa_coarse': 512,
    'multi_eurlex_level_1': 32 * 128,  # 4096
    'multi_eurlex_level_2': 32 * 128,  # 4096
    'multi_eurlex_level_3': 32 * 128,  # 4096
    'online_terms_of_service_clause_topics': 256,
    'online_terms_of_service_unfairness_levels': 256,
    'swiss_judgment_prediction': 16 * 128,  # 2048
}


def get_python_file_for_task(task):
    return f"run_{task}.py"


def generate_command(time_now, **data):
    command_template = 'python3 ./experiments/{CODE} ' \
                       '--model_name_or_path {MODEL_NAME} ' \
                       '--output_dir {LOG_DIRECTORY}/{TASK}/{MODEL_NAME}/seed_{SEED} ' \
                       '--log_directory {LOG_DIRECTORY} ' \
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
        command_template = command_template + ' --dataset_cache_dir {DATASET_CACHE_DIR}'

    if data["language"] is not None:
        command_template = command_template + ' --language {LANGUAGE} '

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
            command_template += ' --fp16 --fp16_full_eval'
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
                   dataset_cache_dir, num_train_epochs=None, log_directory=None):
    # TODO I think it would be easier to just pass the whole data dictionary to the function
    #  so that we only have one parameter and do the same for the generate_command function
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

    if log_directory is None:
        log_directory = 'results/logs_' + str(time_stamp)
    else:
        if os.path.isdir(log_directory) == False:
            os.mkdir(log_directory)

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

    # language_model_type is in the form {type: general|legal}_{language: ISO_CODE or "multilingual"]}_{size: small|base|large}
    language_model_info = language_model_type.split('_')
    if len(language_model_info) != 3:  # we got a direct language model just use it
        models_to_be_used = [language_model_type]
    else:  # find out what models we want to run
        types, languages, sizes = language_model_info[0], language_model_info[1], language_model_info[2]
        if 'all' in types:
            types = _TYPES
        if 'all' in languages:
            languages = _LANGUAGES
        if 'all' in sizes:
            sizes = _SIZES

        # expand comma-separated lists
        types = types.split(',') if ',' in types else [types]
        languages = languages.split(',') if ',' in languages else [languages]
        sizes = sizes.split(',') if ',' in sizes else [sizes]

        models_to_be_used = []
        for t in types:
            for l in languages:
                for s in sizes:
                    if l in language_models[t]:
                        models_to_be_used.extend(language_models[t][l][s])  # add all models we want to run
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
        if language is None:
            if model_name in model_language_lookup_table.keys():
                # all means just the entire dataset, so that we take the default value
                if model_language_lookup_table[model_name] != 'all':
                    language = model_language_lookup_table[model_name]

        script_new = generate_command(
            time_now=time_stamp, gpu_number=gpu_id, gpu_memory=gpu_memory,
            model_name=model_name,
            lower_case=lower_case, task=task, seed=seed,
            num_train_epochs=num_train_epochs, batch_size=batch_size,
            accumulation_steps=accumulation_steps, language=language,
            running_mode=running_mode, learning_rate=learning_rate,
            code=get_python_file_for_task(task), metric_for_best_model=metric_for_best_model,
            hierarchical=hierarchical, greater_is_better=greater_is_better,
            download_mode=download_mode, log_directory=log_directory,
            preprocessing_num_workers=preprocessing_num_workers,
            dataset_cache_dir=dataset_cache_dir
        )
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
                        default=None)
    parser.add_argument('-lc', '--lower_case', help='Define if lower case or not.', default=False)
    parser.add_argument('-lmt', '--language_model_type',
                        help='Define which kind of language model you would like to use (e.g. xlm-roberta-base); '
                             'alternatively, you can format it in the following way:'
                             '{type: general|legal}_{language: ISO_CODE or "multilingual"]}_{size: small|base|large}. '
                             'If you specify "all" for any of the three parameters, '
                             'all models of that type/language/size will be used.',
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
    parser.add_argument('-ld', '--log_directory',
                        help='Specify the directory where you want to save your logs. '
                             'The directory at the end of the tree is used as the project name for wandb.',
                        default='results')
    parser.add_argument('-nw', '--preprocessing_num_workers',
                        help="The number of processes to use for the preprocessing. "
                             "If it deadlocks, try setting this to 1.", default=1)
    parser.add_argument('-cad', '--dataset_cache_dir',
                        help="Specify the directory you want to cache your datasets.",
                        default=None)

    args = parser.parse_args()

    if args.download_mode == 'force_redownload':
        # Remove the existing cache directory since everything will be redownloaded anyway
        # Somehow the cache caused errors
        if args.dataset_cache_dir is not None:
            if os.path.isdir(args.dataset_cache_dir):
                shutil.rmtree(args.dataset_cache_dir)
            else:
                os.mkdir(args.dataset_cache_dir)

    # TODO set sequence length, segment length and segments from here instead of relying on default values
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
        log_directory=args.log_directory,
        preprocessing_num_workers=args.preprocessing_num_workers,
        running_mode=args.running_mode,
        task=args.task
    )
