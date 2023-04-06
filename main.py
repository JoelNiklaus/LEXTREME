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

from utils.utilities import remove_old_files, get_meta_infos, get_hierarchical, get_python_file_for_task, \
    get_optimal_batch_size, get_default_number_of_training_epochs, generate_command

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
        affair_text_scope,
        batch_size,
        dataset_cache_dir,
        download_mode,
        eval_steps,
        evaluation_strategy,
        gpu_memory,
        gpu_number,
        hierarchical,
        text,
        title,
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
        time_stamp = re.sub(':', '_', datetime.datetime.now().isoformat())
        log_directory = log_directory + '/logs_' + str(time_stamp)
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
                affair_text_scope=affair_text_scope,
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
                text=text,
                title=title,
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

    parser.add_argument('-ats', '--affair_text_scope', help="Specify which kanton you would like to include in the dataset.", default='zh,ch')
    parser.add_argument('-as', '--accumulation_steps', help='Define the number of accumulation_steps.', default=None)
    parser.add_argument('-bz', '--batch_size', help='Define the batch size.', default=None)
    parser.add_argument('-es', '--evaluation_strategy',
                        help="The evaluation strategy to adopt during training. Possible values are: no = No "
                             "evaluation is done during training; steps = Evaluation is done (and logged) every "
                             "eval_steps; epoch = Evaluation is done at the end of each epoch.",
                        default=None)
    parser.add_argument('-gn', '--gpu_number', help='Define which GPU you would like to use.', default=None)
    parser.add_argument('-gm', '--gpu_memory', help='Define how much memory your GPUs have', default=11)
    parser.add_argument('-hier', '--hierarchical',
                        help='Define whether you want to use a hierarchical model or not. '
                             'Caution: this will not work for every task',
                        default=None)
    parser.add_argument('-text', '--text',
                        help="Specify if you want to include the text in the training.",
                        default=False, 
                        action="store_true"
                        )
    parser.add_argument('-title', '--title',
                        help="Specify if you want to include the title in the training.",
                        default=False, 
                        action="store_true"
                        )
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
                             'See the options in: '
                             'https://huggingface.co/docs/datasets/v1.5.0/loading_datasets.html#download-mode',
                        default='reuse_dataset_if_exists')  # reuses raw downloaded files but makes dataset freshly
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
        affair_text_scope=args.affair_text_scope,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        dataset_cache_dir=args.dataset_cache_dir,
        download_mode=args.download_mode,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        gpu_memory=args.gpu_memory,
        gpu_number=args.gpu_number,
        hierarchical=args.hierarchical,
        text=args.text,
        title=args.title,
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
