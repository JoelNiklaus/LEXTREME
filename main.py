import argparse
import datetime
import itertools
import json as js
import os
import re
import shutil
import time
from collections import defaultdict
from itertools import cycle
from multiprocessing import Pool
from utils.utilities import remove_old_files

import setproctitle
import torch

from utils.utilities import remove_old_files, get_meta_infos, get_hierarchical, get_python_file_for_task, \
    get_optimal_batch_size, get_default_number_of_training_epochs

# Empty folder with temporary scripts
# shutil.rmtree('./temporary_scripts', ignore_errors=True) # Uncomment if you want to delete all temporary scripts
os.makedirs('./temporary_scripts', exist_ok=True)
remove_old_files('./temporary_scripts')

_TYPES = ['general', 'legal']
_LANGUAGES = ['bg', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'fi', 'fr', 'ga', 'hr',
              'hu', 'it', 'lt', 'lv', 'mt', 'nl', 'pl', 'pt', 'ro', 'sk', 'sl', 'sv']
_SIZES = ['small', 'base', 'large']

# TODO look if there are other models available that we want to run: e.g. small or large ones
# TODO add more legal models maybe from nllpw: https://nllpw.org/workshop/program/
# HERE many models are reported: https://arxiv.org/pdf/2109.00904.pdf

meta_infos = get_meta_infos()


def generate_command(**data):

    time_now = datetime.datetime.now().isoformat()

    if data["hierarchical"] is None:
        data["hierarchical"] = get_hierarchical(data["task"])

    command_template = 'python3 ./experiments/{CODE} ' \
                       '--model_name_or_path {MODEL_NAME} ' \
                       '--log_directory {LOG_DIRECTORY} ' \
                       '--do_train --do_eval --do_predict ' \
                       '--overwrite_output_dir ' \
                       '--load_best_model_at_end --metric_for_best_model {METRIC_FOR_BEST_MODEL} ' \
                       '--greater_is_better {GREATER_IS_BETTER} ' \
                       '--evaluation_strategy {EVALUATION_STRATEGY} ' \
                       '--logging_strategy {LOGGING_STRATEGY} ' \
                       '--save_strategy {SAVE_STRATEGY} ' \
                       '--save_total_limit 6 ' \
                       '--num_train_epochs {NUM_TRAIN_EPOCHS} ' \
                       '--learning_rate {LEARNING_RATE} ' \
                       '--per_device_train_batch_size {BATCH_SIZE} --per_device_eval_batch_size {BATCH_SIZE} ' \
                       '--seed {SEED} ' \
                       '--gradient_accumulation_steps {ACCUMULATION_STEPS} --eval_accumulation_steps {ACCUMULATION_STEPS} ' \
                       '--running_mode {RUNNING_MODE} ' \
                       '--download_mode {DOWNLOAD_MODE} ' \
                       '--preprocessing_num_workers {PREPROCESSING_NUM_WORKERS} ' \
                       '--hierarchical {HIERARCHICAL} ' \
                       '--revision {REVISION} '

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
                                            REVISION=data["revision"]
                                            )

    file_name = './temporary_scripts/' + data["task"] + "_" + str(data["gpu_number"]) + "_" + str(
        data["seed"]) + "_" + str(data["model_name"]).replace('/', '_') + "_" + time_now + ".sh"

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


def run_experiment(
        accumulation_steps,
        batch_size,
        dataset_cache_dir,
        download_mode,
        eval_steps,
        evaluation_strategy,
        gpu_memory,
        gpu_number,
        hierarchical,
        language_model_type,
        learning_rate,
        list_of_languages,
        list_of_seeds,
        logging_strategy,
        logging_steps,
        lower_case,
        preprocessing_num_workers,
        revision,
        running_mode,
        save_strategy,
        save_steps,
        task,
        log_directory=None,
        num_train_epochs=None
):
    # TODO I think it would be easier to just pass the whole data dictionary to the function
    #  so that we only have one parameter and do the same for the generate_command function

    preprocessing_num_workers = int(preprocessing_num_workers)

    batch_size_to_be_found = True if batch_size is None else False
    language_to_be_found = True if list_of_languages is None else False

    if list_of_languages is not None:
        if type(list_of_languages) == str:
            list_of_languages = list_of_languages.split(',') if ',' in list_of_languages else [list_of_languages]

    if log_directory is None:
        log_directory = "results"
        time_stamp = re.sub(':','_',datetime.datetime.now().isoformat())
        log_directory = log_directory+'/logs_' + str(time_stamp)
    else:
        if not os.path.isdir(log_directory):
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
    # TODO remove this for criticality_prediction
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

        # expand comma-separated lists
        types = types.split(',') if ',' in types else [types]
        languages = languages.split(',') if ',' in languages else [languages]
        sizes = sizes.split(',') if ',' in sizes else [sizes]

        # expand 'all' keywords
        if 'all' in types:
            types = _TYPES
        if 'all' in languages:
            languages = _LANGUAGES
        if 'all' in sizes:
            sizes = _SIZES

        models_to_be_used = []
        for t in types:
            for l in languages:
                for s in sizes:
                    if l in meta_infos["language_models"][t]:
                        models_to_be_used.extend(
                            meta_infos["language_models"][t][l][s])  # add all models we want to run

    models_to_be_used = sorted(list(set(models_to_be_used)))
    print(models_to_be_used)

    gpu_command_dict = defaultdict(list)

    metric_for_best_model = "eval_loss"
    greater_is_better = False

    if task == 'all':
        all_variables = [[t for t in list(meta_infos["task_type_mapping"].keys())], models_to_be_used, list_of_seeds]
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

        if list_of_languages is None:
            if model_name in meta_infos["model_language_lookup_table"].keys():
                # all means just the entire dataset, so that we take the default value
                if meta_infos["model_language_lookup_table"][model_name] != 'all':
                    list_of_languages = [meta_infos["model_language_lookup_table"][model_name]]
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

            script_new = generate_command(
                accumulation_steps=accumulation_steps,
                batch_size=batch_size,
                code=get_python_file_for_task(task),
                dataset_cache_dir=dataset_cache_dir,
                download_mode=download_mode,
                evaluation_strategy=evaluation_strategy,
                eval_steps=eval_steps,
                gpu_memory=gpu_memory,
                gpu_number=gpu_id,
                greater_is_better=greater_is_better,
                hierarchical=hierarchical,
                language=lang,
                learning_rate=learning_rate,
                logging_strategy=logging_strategy,
                logging_steps=logging_steps,
                log_directory=log_directory,
                lower_case=lower_case,
                metric_for_best_model=metric_for_best_model,
                model_name=model_name,
                num_train_epochs=num_train_epochs,
                preprocessing_num_workers=preprocessing_num_workers,
                revision=revision,
                running_mode=running_mode,
                save_strategy=save_strategy,
                save_steps=save_steps,
                seed=seed,
                task=task
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
    parser.add_argument('-es', '--evaluation_strategy',
                        help="The evaluation strategy to adopt during training. Possible values are: no = No "
                             "evaluation is done during training; steps = Evaluation is done (and logged) every "
                             "eval_steps; epoch = Evaluation is done at the end of each epoch.",
                        default=None)
    parser.add_argument('-gn', '--gpu_number', help='Define which GPU you would like to use.', default=None)
    parser.add_argument('-gm', '--gpu_memory', help='Define how much memory your GPUs have', default=None)
    parser.add_argument('-hier', '--hierarchical',
                        help='Define whether you want to use a hierarchical model or not. '
                             'Caution: this will not work for every task',
                        default=None)
    parser.add_argument('-ls', '--logging_strategy',
                        help="The logging strategy to adopt during training. Possible values are: no: No logging is "
                             "done during training ; epoch: Logging is done at the end of each epoch; steps: Logging "
                             "is done every logging_steps.",
                        default=None)
    parser.add_argument('-lol', '--list_of_languages',
                        help='Define if you want to filter the training dataset by language.',
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
    parser.add_argument('-rev', '--revision', help='The specific model version to use. It can be a branch name, '
                                                   'a tag name, or a commit id, since we use a git-based system for '
                                                   'storing models and other artifacts on huggingface.co, so revision '
                                                   'can be any identifier allowed by git.', default="main")
    parser.add_argument('-rmo', '--running_mode',
                        help='Define whether you want to run the finetuning on all available training data ("default"), '
                             'on 100 examples for debugging ("debug") '
                             'or on 5% for testing purposes ("experimental").',
                        default='default')
    parser.add_argument('-dmo', '--download_mode',
                        help='Define whether you want to redownload the dataset or not. '
                             'See the options in: https://huggingface.co/docs/datasets/v1.5.0/loading_datasets.html'
                             '#download-mode',
                        default='reuse_cache_if_exists')  # reuses raw downloaded files but makes dataset freshly
    parser.add_argument('-t', '--task', help='Choose a task.', default='all',
                        choices=sorted(list(meta_infos["task_type_mapping"].keys())))
    parser.add_argument('-ld', '--log_directory',
                        help='Specify the directory where you want to save your logs. '
                             'The directory at the end of the tree is used as the project name for wandb.',
                        default=None)
    parser.add_argument('-nw', '--preprocessing_num_workers',
                        help="The number of processes to use for the preprocessing. "
                             "If it deadlocks, try setting this to 1.", default=1)
    parser.add_argument('-cad', '--dataset_cache_dir',
                        help="Specify the directory you want to cache your datasets.",
                        default=None)
    parser.add_argument('-ss', '--save_strategy',
                        help="The checkpoint save strategy to adopt during training. Possible values are: no: No save "
                             "is done during training; epoch: Save is done at the end of each epoch; steps: Save is "
                             "done every save_steps.",
                        default=None)
    parser.add_argument('-est', '--eval_steps',
                        help='Number of update steps between two evaluations if evaluation_strategy="steps". Will '
                             'default to the same value as logging_steps if not set.',
                        default=None)
    parser.add_argument('-lst', '--logging_steps',
                        help='Number of update steps between two logs if logging_strategy="steps".', default=None)
    parser.add_argument('-sst', '--save_steps',
                        help='Number of updates steps before two checkpoint saves if save_strategy="steps".',
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
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        gpu_memory=args.gpu_memory,
        gpu_number=args.gpu_number,
        hierarchical=args.hierarchical,
        list_of_languages=args.list_of_languages,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        language_model_type=args.language_model_type,
        learning_rate=args.learning_rate,
        list_of_seeds=args.list_of_seeds,
        lower_case=args.lower_case,
        num_train_epochs=args.num_train_epochs,
        log_directory=args.log_directory,
        preprocessing_num_workers=args.preprocessing_num_workers,
        revision=args.revision,
        running_mode=args.running_mode,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        task=args.task
    )
