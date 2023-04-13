import argparse
import os
import shutil
from utils.utilities import remove_old_files, get_meta_infos, run_experiment, make_boolean

# Empty folder with temporary scripts
# shutil.rmtree('./temporary_scripts', ignore_errors=True) # Uncomment if you want to delete all temporary scripts
os.makedirs('./temporary_scripts', exist_ok=True)
remove_old_files('./temporary_scripts')

# TODO look if there are other models available that we want to run: e.g. small or large ones
# TODO add more legal models maybe from nllpw: https://nllpw.org/workshop/program/
# HERE many models are reported: https://arxiv.org/pdf/2109.00904.pdf

meta_infos = get_meta_infos()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-as', '--accumulation_steps', help='Define the number of accumulation_steps.', default=None)
    parser.add_argument('-bz', '--batch_size', help='Define the batch size.', default=None)
    parser.add_argument('-dhs', '--do_hyperparameter_search',
                        help="Specify if you want to apply hyperparameter search.", default=False)
    parser.add_argument('-es', '--evaluation_strategy',
                        help="The evaluation strategy to adopt during training. Possible values are: no = No "
                             "evaluation is done during training; steps = Evaluation is done (and logged) every "
                             "eval_steps; epoch = Evaluation is done at the end of each epoch.",
                        default=None)
    parser.add_argument('-gn', '--gpu_number', help='Define which GPU you would like to use.', default=None)
    parser.add_argument('-gib', '--greater_is_better',
                        help='Use in conjunction with load_best_model_at_end and metric_for_best_model to specify if better models should have a greater metric or not. Will default to: True if metric_for_best_model is set to a value that isn’t "loss" or "eval_loss". False if metric_for_best_model is not set, or set to "loss" or "eval_loss". ',
                        default=None)
    parser.add_argument('-gm', '--gpu_memory', help='Define how much memory your GPUs have', default=11)
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
    parser.add_argument('-lr', '--learning_rate', help='Define the learning rate.', default=1e-5)
    parser.add_argument('-mfbm', '--metric_for_best_model',
                        help='Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix "eval_". Will default to "loss" if unspecified and load_best_model_at_end=True (to use the evaluation loss). If you set this value, greater_is_better will default to True. Don’t forget to set it to False if your metric is better when lower. ',
                        default="eval_loss")
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
                        choices=['all'] + sorted(list(meta_infos["task_type_mapping"].keys())))
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
        do_hyperparameter_search=make_boolean(args.do_hyperparameter_search),
        download_mode=args.download_mode,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        greater_is_better=args.greater_is_better,
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
        metric_for_best_model=args.metric_for_best_model,
        num_train_epochs=args.num_train_epochs,
        log_directory=args.log_directory,
        preprocessing_num_workers=args.preprocessing_num_workers,
        revision=args.revision,
        running_mode=args.running_mode,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        task=args.task
    )
