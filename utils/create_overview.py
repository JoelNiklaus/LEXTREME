from datetime import datetime
from pathlib import Path
from statistics import mean
import argparse
import logging
import os
import pandas as pd
import re
import sys
import numpy as np
from collections import defaultdict
from copy import deepcopy
import wandb
from traceback import print_exc
from utilities import get_meta_infos
import json as js

joels_models = pd.read_excel(os.path.join(
    os.path.dirname(__file__), 'joels_models.xlsx'))
models_currently_to_be_ignored = joels_models[
    joels_models.need_to_be_run == "no"]._name_or_path.tolist()


def insert_revision(_name_or_path):
    if _name_or_path in joels_models._name_or_path.tolist():
        return joels_models[joels_models._name_or_path == _name_or_path].revision.tolist()[0]
    else:
        return 'main'


meta_infos = get_meta_infos()


def insert_responsibilities(df):
    respons = pd.read_excel(os.path.join(
        os.path.dirname(__file__), 'run_responsibility.xlsx'))
    respons = respons[['finetuning_task',
                       '_name_or_path', 'responsible', 'gpu']]
    df_merged = pd.merge(df, respons, how='left', right_on=[
        'finetuning_task', '_name_or_path'], left_on=['finetuning_task', '_name_or_path'])
    return df_merged


class ResultAggregator:
    meta_infos = meta_infos

    def __init__(self,
                 only_completed_tasks=False,
                 path_to_csv_export=None,
                 project_name=None,
                 score='macro-f1',
                 split="predict",
                 verbose_logging=True,
                 wandb_api_key=None,
                 which_language=None,
                 ):
        """
        @type only_completed_tasks: bool
        @type path_to_csv_export: str
        @type project_name: str
        @type score: Must be one of these: [macro-f1, macro-precision, macro-recall, micro-f1, micro-precision, micro-recall, weighted-f1, weighted-precision, weighted-recall]
        @type split: Must be one of these: [train, validation, test]
        @verbose_logging: bool
        @wandb_api_key: str
        @which_language: str

        project_name: Default value is None. Specify the name of the project from wand which you want to fetch your data from.
        Do not forget to put the name of the wandb team in fron of your project name.
        For example: if your wand team name is awesome_team and your project tame awseome_project, you need to pass awesome_team/awesome_project
        If set to None, it will create csv file called `current_wandb_results_unprocessed.csv` in your current directory.
        If set to None, it will presumably take the api key which you are already logged with.
        If set to None, it will presumably take the last project that you used with the wandb api.

        only_completed_tasks: Default value is False. Specify if you want to consider only tasks in the reports that are entirely completed. Entirely completed means that all predefined models have successuflly run for all seeds. It is advised to keep it False.
        path_to_csv_export: Default value is None. Specify the path and name of the csv file where you would liek to export your wandb results to save them locally.

        score: Default value is macro-f1. Specify the default score which will be used to create reports.

        split: Default value is `predict`. Specify the split which want to use to create your reports.

        verbose_logging: Default value is True. Specify if you want print out all logging information. If set to False, the logs will not be printed out. However, all logging information will always be saved in a new file unter the logs folder.

        wandb_api_key: Default value is None. Insert the wandb api key for the project you are interested in.

        which_language: Default value is None. Choose which language should be considered for the overviews. If set to `monolingual`, only monolingual models will be considered. If set to `multilingual`, only multilingual models will be considered. Otherwise, only those models will be considered that belong to the defined language. For example, if you pass `de`, only German models will be considered.

        """
        # Create the result directory if not existent
        if os.path.isdir('results') == False:
            os.mkdir('results')

        if split == "eval":
            self.split = split + "/"
        elif split == "predict":
            self.split = split + "/_"

        self.score = score

        self.api = wandb.Api()

        if os.path.exists('logs') == False:
            os.mkdir('logs')

        name_of_log_file = 'logs/loggings_' + datetime.now().isoformat() + '.txt'
        name_of_log_file = os.path.join(
            os.path.dirname(__file__), name_of_log_file)

        self.only_completed_tasks = only_completed_tasks

        self.which_language = which_language

        if verbose_logging:
            handlers = [
                logging.FileHandler(name_of_log_file),
                logging.StreamHandler()
            ]
        elif verbose_logging == False:
            handlers = [
                logging.FileHandler(name_of_log_file)
            ]

        logging.basicConfig(level=logging.DEBUG,
                            handlers=handlers,
                            format="%(asctime)-15s %(levelname)-8s %(message)s"
                            )

        self.wandb_api_key = wandb_api_key

        if type(wandb_api_key) == str and bool(re.search('\d', wandb_api_key)):
            os.environ["WANDB_API_KEY"] = self.wandb_api_key

        self._path = name_of_log_file
        if path_to_csv_export is None:
            results = self.get_wandb_overview(project_name)
            results.to_csv(
                "current_wandb_results_unprocessed.csv", index=False)
        else:
            results = pd.read_csv(Path(path_to_csv_export))
            results = self.edit_result_dataframe(results, name_editing=False)
        results = results[results.finetuning_task.isnull() == False]

        self.results = results

        available_predict_scores = [col for col in self.results if bool(
            re.search(r'^\w+_predict/_' + self.score, col))]
        self.available_predict_scores = sorted(available_predict_scores)
        self.available_predict_scores_original = deepcopy(
            self.available_predict_scores)

        # Add column to rows to indicate if this run pertains to completed tasks or not
        self.mark_incomplete_tasks()

    def __enter__(self):
        sys.stdout = open(self._path, mode="w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def get_info(self):
        logging.info(
            "The current export contains results for the following tasks:")
        logging.info(self.results.finetuning_task.unique())

    def remove_slashes(self, string):
        return re.sub('\/', '', string)

    def generate_concrete_score_name(self):

        languages = {'hu', 'en', 'sv', 'da', 'nl', 'pl', 'de', 'mt', 'it', 'cs', 'ro', 'lt',
                     'sk', 'hr', 'fi', 'nb', 'lv', 'el', 'bg', 'pt', 'et', 'sl', 'fr', 'es', 'ga'}
        scores = ['macro-f1', 'micro-f1', 'weighted-f1', 'macro-precision', 'macro-recall', 'micro-recall',
                  'weighted-recall',
                  'macro-precision', 'micro-precision', 'weighted-precision', 'accuracy_normalized']
        splits = ['eval/', 'predict/_']

        score_names = set()

        for spl in splits:
            for lang in languages:
                for s in scores:
                    if 'predict' in spl:
                        score_name = lang + '_' + spl + s
                        score_names.add(score_name)
                    score_name = spl + s
                    score_names.add(score_name)

        return score_names

    def fetch_data(self, project_name):

        # Project is specified by <entity/project-name>
        runs = self.api.runs(project_name)

        score_names = self.generate_concrete_score_name()

        score_names.add('eval/loss')
        score_names.add('predict/_loss')

        results = list()
        for x in runs:
            entry = dict()
            try:
                if x.state == "finished":
                    entry['state'] = x.state
                    entry["finetuning_task"] = x.config['finetuning_task']
                    entry["seed"] = x.config['seed']
                    entry["_name_or_path"] = x.config['_name_or_path']
                    entry['name'] = x.name
                    for sn in score_names:
                        if sn in x.history_keys['keys'].keys():
                            entry[sn] = x.history_keys['keys'][sn]['previousValue']
                        else:
                            entry[sn] = ''
                    results.append(entry)
            except:
                print_exc()

        return results

    def get_wandb_overview(self, project_name: str = None):
        if project_name is None:
            project_name = "lextreme/paper_results"
        # results = wandb_summarizer.download.get_results(project_name=project_name)
        results = self.fetch_data(project_name=project_name)
        results = pd.DataFrame(results)
        results.to_csv(
            "results/current_wandb_results_unprocessed.csv", index=False)
        results = self.edit_result_dataframe(results)
        return results

    def edit_column_name(self, column_name):
        if column_name.startswith('config_'):
            column_name = re.sub('^config_', '', column_name)
        if column_name.startswith('end'):
            column_name = re.sub('^end_', '', column_name)
        return column_name

    def insert_hierarchical(self, run_name: str):
        "Infos whether the run was hierarchical were not logged to wandb in the beginning. But this information is in the name of the run. This function will extract this information."

        if 'hierarchical_True' in run_name:
            return True
        elif 'hierarchical_False' in run_name:
            return False

    def edit_column_names_in_df(self, dataframe):
        columns = dataframe.columns.tolist()

        for col in columns:
            col_new = self.edit_column_name(col)
            dataframe = dataframe.rename(columns={col: col_new})

        return dataframe

    def update_language_specifc_predict_columns(self, dataframe, score=None):
        """ For multilinugal datasets we insert the overall macro-f1 score into the column for the language specific macro-f1 score if a monolingual model was finetuned on them. """

        if score is None:
            score = self.split + self.score

        monolingual_configs = []
        multilingual_configs = []

        for config, languages in self.meta_infos['task_language_mapping'].items():
            if len(languages) == 1:
                monolingual_configs.append(config)
            else:
                multilingual_configs.append(config)

        all_available_languages_for_training_set = list(
            dataframe['language'].unique())
        all_available_languages_for_training_set = [
            l for l in all_available_languages_for_training_set if l != "all"]

        for config in monolingual_configs:
            languages = self.meta_infos['task_language_mapping'][config]
            for language in languages:
                column_name = language + '_' + score
                indices = dataframe[
                    (dataframe.language == language) & (dataframe.finetuning_task == config)].index.tolist()
                for i in indices:
                    dataframe.at[i, column_name] = dataframe.at[i, score]

        monolingual_models = [_model_name for _model_name, _language in
                              self.meta_infos['model_language_lookup_table'].items()
                              if _language != "all"]

        for config in multilingual_configs:
            languages = self.meta_infos['task_language_mapping'][config]
            for language in languages:
                column_name = language + '_' + score
                indices = dataframe[(dataframe.language == language) & (dataframe.finetuning_task == config) & (
                    dataframe._name_or_path.isin(monolingual_models))].index.tolist()
                for i in indices:
                    if type(dataframe.at[i, column_name]) != float:
                        dataframe.at[i, column_name] = dataframe.at[i, score]

        return dataframe

    def language_and_model_match(self, language_model, language):

        if self.meta_infos["model_language_lookup_table"][language_model] == "all":
            return True
        else:
            return self.meta_infos["model_language_lookup_table"][language_model] == language

    def correct_language(self, finetuning_task, language_model, language):
        '''
        Sometimes the language column is wrong.
        For example, for greek_legal_code_subject_level in combination with distilbert-base-multilingual-cased,
        the run names start with “all_” instead of “el_” which should not be.
        This code will check if a language model is multilingual and if a task is monolingual.
        If the language is all, it must be corrected to the language of the monolingual model.
        '''
        if self.meta_infos["model_language_lookup_table"][language_model] == "all":
            task_lang_mapping = self.meta_infos["task_language_mapping"]
            if finetuning_task in task_lang_mapping and len(task_lang_mapping[finetuning_task]) == 1:
                if language == "all":
                    new_language = task_lang_mapping[finetuning_task][0]
                    log_message = ' '.join(["Changed language to from " + language + " to " + new_language +
                                            " for the following combinations: ", finetuning_task, language_model])
                    logging.info(log_message)
                    return new_language
        return language

    def check_for_model_language_consistency(self, dataframe):

        dataframe['language'] = dataframe.apply(
            lambda x: self.correct_language(x["finetuning_task"], x["_name_or_path"], x["language"]), axis=1)

        dataframe['model_language_consistency'] = dataframe.apply(
            lambda x: self.language_and_model_match(x["_name_or_path"], x["language"]), axis=1)

        dataframe = dataframe[dataframe.model_language_consistency == True]
        del dataframe["model_language_consistency"]

        return dataframe

    def loss_equals_nan(self, loss):
        '''
        For some reason, when wandb shows loss == nan, you will not get a nan value from the API.
        Instead, it returns a very low value in scientifc notation.
        This functions tries to detect these cases. 
        A manuel evaluation showed that the predictions are worse in these cases, so we need to rerun the experiments again.

        '''

        return '+' in str(loss)

    def edit_result_dataframe(self, results, name_editing=True):
        results = self.edit_column_names_in_df(results)
        results = results[results._name_or_path.str.contains(
            'BSC-TeMU/roberta-base-bne') == False]  # This model was removed later because it was not available on huggingface in a later stage during the project
        # all_finetuning_tasks = results.finetuning_task
        if name_editing == True:
            results['language'] = results.finetuning_task.apply(
                lambda x: str(x).split('_')[0])
            results['finetuning_task'] = results.finetuning_task.apply(
                lambda x: '_'.join(str(x).split('_')[1:]))
        results['hierarchical'] = results.name.apply(self.insert_hierarchical)
        # Remove all cases where language == nan
        results = results[results.language != "nan"]
        results['seed'] = results.seed.apply(lambda x: int(x))
        results = self.update_language_specifc_predict_columns(results)
        results = self.check_for_model_language_consistency(results)

        # Remove all cases where eval/loss is not a float
        results = results[results['eval/loss'] != ""]

        # Remove all cases where predict/_loss is not a float
        results = results[results['predict/_loss'] != ""]

        # Remove all cases where eval/loss is nan according to wandb
        results = results[results['eval/loss']
                          .apply(self.loss_equals_nan) == False]

        # Keep only results from seed 1,2,3 and remove anything else
        results = results[results.seed.isin([1, 2, 3])]

        # When fetching the data from the wandb api it it return state instead of State as column name
        # I could make every column name lowercase, but I am afraid this might cause some problems
        if 'State' in results.columns:
            results = results[results.State == "finished"]
        elif 'state' in results.columns:
            results = results[results.state == "finished"]

        results = results.drop_duplicates(
            ["seed", "finetuning_task", "_name_or_path", "language"])

        # TODO: Remove this constraint in the future
        # results = results[results._name_or_path.str.contains('joelito') == False]
        results = results[
            results.finetuning_task.str.contains('turkish_constitutional_court_decisions_judgment') == False]

        return results

    def remove_languages(self, languages):
        if type(languages) == str:
            self.available_predict_scores = [score for score in self.available_predict_scores if
                                             score.startswith(languages) == False]
        elif type(languages) == list:
            for lang in languages:
                self.available_predict_scores = [score for score in self.available_predict_scores if
                                                 score.startswith(lang) == False]

    def reset_list_of_available_languages(self):
        self.available_predict_scores = deepcopy(
            self.available_predict_scores_original)

    def filter_by_language(self, dataframe, which_language):
        if which_language == 'monolingual':
            dataframe = dataframe[
                dataframe._name_or_path.apply(lambda x: meta_infos['model_language_lookup_table'][x] != 'all')]
        elif which_language == 'multilingual':
            dataframe = dataframe[
                dataframe._name_or_path.apply(lambda x: meta_infos['model_language_lookup_table'][x] == 'all')]
        else:
            dataframe = dataframe[
                dataframe._name_or_path.apply(
                    lambda x: meta_infos['model_language_lookup_table'][x] == which_language)]
        return dataframe

    def check_seed_per_task(self, tasks_to_filter=[], which_language=None):

        if which_language is None:
            which_language = self.which_language

        report = list()

        required_seeds = {"1", "2", "3"}

        for task, languages in self.meta_infos["task_language_mapping"].items():
            required_models = list()

            for model_name, language in self.meta_infos['model_language_lookup_table'].items():
                if language == "all":
                    required_models.append(model_name)
                else:
                    for lang in languages:
                        if lang == language:
                            required_models.append(model_name)

            available_models = self.results[(self.results.finetuning_task == task) & (
                self.results._name_or_path.isin(required_models))]._name_or_path.unique()

            for rm in required_models:
                if rm not in available_models:
                    message = "For task " + task + " in combination with the language " + \
                              self.meta_infos['model_language_lookup_table'][
                                  rm] + " we do not have any results for this model: " + rm
                    logging.warning(message)
                    item = dict()
                    item['finetuning_task'] = task
                    item['_name_or_path'] = rm
                    item['language'] = self.meta_infos['model_language_lookup_table'][rm]
                    item['missing_seeds'] = sorted(list(required_seeds))
                    report.append(item)

            for am in available_models:
                list_of_seeds = set(self.results[(self.results.finetuning_task == task) & (
                        self.results._name_or_path == am)].seed.unique())
                list_of_seeds = set([str(int(x)) for x in list_of_seeds])
                if list_of_seeds.intersection(required_seeds) != required_seeds:
                    missing_seeds = set(
                        [s for s in required_seeds if s not in list_of_seeds])
                    message = "There not enough seeds for task " + task + " in combination with the language model " + am + ". We have only results for the following seeds: ", ', '.join(
                        list(list_of_seeds)) + ". Results are missing for the following seeds: ", ', '.join(
                        list(missing_seeds))
                    logging.warning(message)
                    item = dict()
                    item['finetuning_task'] = task
                    item['_name_or_path'] = am
                    item['language'] = self.meta_infos['model_language_lookup_table'][am]
                    item['missing_seeds'] = sorted(list(missing_seeds))
                    report.append(item)

        report_df = pd.DataFrame(report)
        report_df['missing_seeds'] = report_df.missing_seeds.apply(
            lambda x: ','.join(x))

        report_df = report_df.drop_duplicates()

        # TODO Remove that later
        report_df = report_df[report_df._name_or_path.isin(
            models_currently_to_be_ignored) == False]
        # report_df = report_df[report_df.finetuning_task != "turkish_constitutional_court_decisions_judgment"]
        if which_language is not None:
            report_df = self.filter_by_language(report_df, which_language)

        report_df['revision'] = report_df._name_or_path.apply(insert_revision)

        report_df = insert_responsibilities(report_df)

        if len(tasks_to_filter) > 0:
            report_df = report_df[report_df.finetuning_task.isin(tasks_to_filter)]

        return report_df

    def mark_incomplete_tasks(self):

        seed_check = self.check_seed_per_task()

        seed_check["look_up"] = seed_check["finetuning_task"] + "_" + \
                                seed_check["_name_or_path"] + "_" + seed_check["language"]

        self.results["look_up"] = self.results["finetuning_task"] + \
                                  "_" + self.results["_name_or_path"] + \
                                  "_" + self.results["language"]

        self.results['completed_task'] = np.where(
            self.results.look_up.isin(seed_check["look_up"].tolist()), False, True)

        del self.results['look_up']

    def create_overview_of_results_per_seed(self, score=None, only_completed_tasks=None, which_language=None):

        if only_completed_tasks is None:
            only_completed_tasks = self.only_completed_tasks

        if which_language is None:
            which_language = self.which_language

        old_score = ""

        if score is None:
            score = self.score
        else:
            old_score = deepcopy(self.score)
            self.score = score

        seed_check = self.check_seed_per_task(which_language=which_language)

        incomplete_tasks = set(seed_check.finetuning_task.unique())

        required_seeds = [1, 2, 3]

        if bool(re.search('(eval|train)', score)):
            score_to_filter = score
        else:
            score_to_filter = self.split + score

        if only_completed_tasks == True:
            # Caution! If there is no completed task, it can happen that the resulting dataframe is empty!
            df = self.results[(self.results.seed.isin(required_seeds)) & (
                    self.results.finetuning_task.isin(incomplete_tasks) == False)][
                ['finetuning_task', '_name_or_path', 'language', 'seed', score_to_filter]]
        elif only_completed_tasks == False:
            df = self.results[(self.results.seed.isin(required_seeds))][
                ['finetuning_task', '_name_or_path', 'language', 'seed', score_to_filter]]

        df_pivot = df.pivot_table(values=score_to_filter, index=[
            'finetuning_task', '_name_or_path', 'language'], columns='seed', aggfunc='first')
        datasets = [self.meta_infos['config_to_dataset'][i[0]]
                    for i in df_pivot.index.tolist()]
        df_pivot['dataset'] = datasets
        df_pivot.reset_index(inplace=True)
        df_pivot['task_type'] = df_pivot["finetuning_task"].apply(
            lambda x: self.meta_infos["task_type_mapping"][x])

        # First wee need to check if there are all seed values as columns
        # As stated in the above comment; if there are no completed tasks, there will not be any seeds values left
        df_pivot_columns = df_pivot.columns.tolist()
        if df_pivot.shape[0] == 0:
            for seed in [1, 2, 3]:
                if seed not in df_pivot_columns:
                    df_pivot[seed] = ''

        df_pivot = df_pivot[['dataset', 'finetuning_task', 'task_type',
                             '_name_or_path', 'language', 1, 2, 3]]
        df_pivot['mean_over_seeds'] = df_pivot.mean(axis=1, numeric_only=True)
        df_pivot['mean_over_seeds'] = df_pivot.mean(axis=1, numeric_only=True)

        if old_score:
            self.score = old_score

        df_pivot['standard_deviation'] = df_pivot[[1, 2, 3]].std(axis=1)

        if which_language is not None:
            df_pivot = self.filter_by_language(df_pivot, which_language)

        return df_pivot

    def create_report(self, only_completed_tasks=None, which_language=None, tasks_to_filter=[]):

        if only_completed_tasks is None:
            only_completed_tasks = self.only_completed_tasks

        if which_language is None:
            which_language = self.which_language

        seed_check = self.check_seed_per_task(tasks_to_filter=tasks_to_filter)
        self.seed_check = seed_check
        macro_f1_overview = self.create_overview_of_results_per_seed(score="macro-f1",
                                                                     only_completed_tasks=only_completed_tasks,
                                                                     which_language=which_language
                                                                     )
        self.macro_f1_overview = macro_f1_overview
        micro_f1_overview = self.create_overview_of_results_per_seed(score="micro-f1",
                                                                     only_completed_tasks=only_completed_tasks,
                                                                     which_language=which_language
                                                                     )
        self.micro_f1_overview = micro_f1_overview
        weighted_f1_overview = self.create_overview_of_results_per_seed(score="weighted-f1",
                                                                        only_completed_tasks=only_completed_tasks,
                                                                        which_language=which_language
                                                                        )
        self.weighted_f1_overview = weighted_f1_overview
        accuracy_normalized_overview = self.create_overview_of_results_per_seed(score="accuracy_normalized",
                                                                                only_completed_tasks=only_completed_tasks,
                                                                                which_language=which_language
                                                                                )
        self.accuracy_normalized_overview = accuracy_normalized_overview

        eval_macro_f1_overview = self.create_overview_of_results_per_seed(
            score="eval/macro-f1", only_completed_tasks=only_completed_tasks, which_language=which_language)

        self.eval_macro_f1_overview = eval_macro_f1_overview

        # write to csv, so we can read it with a bash script
        self.seed_check.to_csv("results/completeness_report.tsv", sep="\t", index=False)

        with pd.ExcelWriter('results/report.xlsx') as writer:
            split_name = self.remove_slashes(self.split)
            self.seed_check.to_excel(
                writer, index=False, sheet_name="completeness_report")
            self.macro_f1_overview.to_excel(
                writer, index=False, sheet_name=split_name + "_macro_f1_overview")
            self.micro_f1_overview.to_excel(
                writer, index=False, sheet_name=split_name + "_micro_f1_overview")
            self.weighted_f1_overview.to_excel(
                writer, index=False, sheet_name=split_name + "_weighted_f1_overview")
            self.accuracy_normalized_overview.to_excel(
                writer, index=False, sheet_name=split_name + "_accuracy_normalized_overview")
            self.eval_macro_f1_overview.to_excel(
                writer, index=False, sheet_name="eval_macro__f1_overview")

    def convert_numpy_float_to_python_float(self, value):
        if value == np.nan:
            final_value = ""
        elif isinstance(value, np.floating):
            final_value = value.item()
        else:
            final_value = value

        return final_value

    def get_mean_from_list_of_values(self, list_of_values):
        list_of_values = [x for x in list_of_values if type(x) != str]
        if len(list_of_values) > 0:
            return self.convert_numpy_float_to_python_float(mean(list_of_values))
        else:
            return ""

    def get_average_score(self, finetuning_task: str, _name_or_path: str, score: str = None) -> float:

        if score is None:
            score = 'predict/_' + self.score

        results_filtered = \
            self.results[
                (self.results.finetuning_task == finetuning_task) & (self.results._name_or_path == _name_or_path)][
                ["seed", score]]

        if len(results_filtered.seed.unique()) < 3:
            logging.warning(
                "Attention. For task " + finetuning_task + " with model name " + _name_or_path + " you have " + str(
                    len(results_filtered.seed.unique())) + " instead of 3 seeds! The mean will be calculated anyway.")

        if len(results_filtered.seed.unique()) != len(results_filtered.seed.tolist()):
            logging.warning(
                'Attention! It seems you have duplicate seeds for task ' + finetuning_task + ' with the model ' + _name_or_path)
            logging.info(
                'Duplicate values will be removed based on the seed number.')

        if len(results_filtered.seed.tolist()) == 0:
            logging.info('It seems you have duplicate seeds task ' + finetuning_task +
                         ' with the model ' + _name_or_path + " has no meaningful results.")
            return ""  # There is nothing to be calculated
        else:

            mean_value = pd.to_numeric(results_filtered[score]).mean()

            mean_value = self.convert_numpy_float_to_python_float(mean_value)

            if mean_value in ["", np.nan]:
                print('There is an error for ',
                      finetuning_task, _name_or_path, score)
                print(results_filtered[score].tolist())

        return mean_value

    def create_template(self, columns=None):
        # Create empty dataframe
        overview_template = pd.DataFrame()

        # Se the list of model names as index
        overview_template = overview_template.set_axis(
            self.results._name_or_path.unique())

        # Name the index axis as Model
        overview_template.index.names = ['Model']

        if columns is None:
            for finetuning_task in self.results.finetuning_task.unique():
                overview_template[finetuning_task] = ''
        else:
            for col in columns:
                overview_template[col] = ''

        # Sort the models according to language
        x = self.meta_infos['model_language_lookup_table']
        models_sorted = list(x.keys())
        overview_template = overview_template.reindex(index=models_sorted)

        return overview_template

    def insert_aggregated_score_over_language_models(self, dataframe, column_name="aggregated_score"):

        dataframe[column_name] = ""

        for _name_or_path in dataframe.index.tolist():
            columns = dataframe.columns.tolist()
            all_mean_macro_f1_scores = dataframe.loc[_name_or_path].tolist()
            all_mean_macro_f1_scores_cleaned = [
                x for x in all_mean_macro_f1_scores if type(x) != str]
            string_values_indices = [columns[all_mean_macro_f1_scores.index(x)] for x in all_mean_macro_f1_scores if
                                     type(x) == str]
            string_values_indices = list(set(string_values_indices))
            if all_mean_macro_f1_scores_cleaned != all_mean_macro_f1_scores:
                logging.warning(
                    'Attention! ' + _name_or_path + ' has string values as mean score for the following datasets/languages: ' + ', '.join(
                        string_values_indices))

            all_mean_macro_f1_scores_mean = self.get_mean_from_list_of_values(
                all_mean_macro_f1_scores_cleaned)
            dataframe.at[_name_or_path, column_name] = all_mean_macro_f1_scores_mean

        columns = sorted(dataframe.columns.tolist())
        first_column = [column_name]
        dataframe = dataframe[first_column +
                              [col for col in columns if col not in first_column]]

        # Check if column with aggregated scores contains only numerical values
        # If yes, we can sort the column
        # if len([x for x in dataframe[column_name].tolist() if type(x)==float])==len(dataframe[column_name].tolist()):
        # dataframe = dataframe.sort_values(column_name, ascending = False)

        return dataframe

    def insert_config_average_scores(self, overview_template, average_over_language=True):

        logging.info("*** Calculating average scores ***")

        if average_over_language == False:
            for _name_or_path in self.results._name_or_path.unique():
                for finetuning_task in self.results.finetuning_task.unique():
                    mean_macro_f1_score = self.get_average_score(
                        finetuning_task, _name_or_path)
                    overview_template.at[_name_or_path, finetuning_task] = mean_macro_f1_score

        elif average_over_language == True:
            for _name_or_path in self.results._name_or_path.unique():
                for finetuning_task in self.results.finetuning_task.unique():

                    # We check of the finetuning task has more than one language
                    # If not, we can process with the normal predict/_macro-f1 that stands for the entire config
                    # If yes, we loop through all avalaible language-specific macro-f1 scores

                    if len(self.meta_infos["task_language_mapping"][finetuning_task]) == 1:
                        mean_macro_f1_score = self.get_average_score(
                            finetuning_task, _name_or_path)

                    elif len(self.meta_infos["task_language_mapping"][finetuning_task]) > 1:
                        predict_language_mean_collected = list()
                        predict_language_collected = list()

                        # Average over languages
                        for aps in self.available_predict_scores:
                            language = aps.split('_')[0]
                            if language in self.meta_infos["task_language_mapping"][finetuning_task]:
                                if self.meta_infos["model_language_lookup_table"][_name_or_path] == language or \
                                        self.meta_infos["model_language_lookup_table"][_name_or_path] == 'all':
                                    # print(finetuning_task, _name_or_path, aps)
                                    predict_language_mean = self.get_average_score(
                                        finetuning_task, _name_or_path, aps)
                                    # This is to avoid string values; if there were no scores available I returned an empty string, because 0.0 would be missleading
                                    if predict_language_mean not in ["", np.nan]:
                                        predict_language_mean_collected.append(
                                            predict_language_mean)
                                        predict_language_collected.append(
                                            language)
                                    # else:
                                    # TODO: Add logger informartion

                        if len(predict_language_mean_collected) > 0:

                            mean_macro_f1_score = self.get_mean_from_list_of_values(
                                predict_language_mean_collected)

                            if set(predict_language_collected) != set(
                                    self.meta_infos["task_language_mapping"][finetuning_task]):
                                if set(self.meta_infos["model_language_lookup_table"][_name_or_path]) != set(
                                        predict_language_collected):
                                    logging.error(
                                        "We do not have the prediction results for all languages for finetuning task " + finetuning_task + " with language model " + _name_or_path)
                                    logging.info("We have results for the following languages: " + ", ".join(
                                        sorted(predict_language_collected)))
                                    logging.info("But we need results for the following languages: " + ", ".join(
                                        sorted(self.meta_infos["task_language_mapping"][finetuning_task])))
                                    logging.info(
                                        "The the results for the following language(s) are mssing: " + ', '.join(
                                            [l for l in self.meta_infos["task_language_mapping"][finetuning_task] if
                                             l not in predict_language_collected]))

                        else:
                            mean_macro_f1_score = ""

                    overview_template.at[_name_or_path, finetuning_task] = mean_macro_f1_score

        # if overview_template.isnull().values.any():
        # logging.warning('Attention! For some cases we do not have an aggregated score! These cases will be converted to nan.')
        # overview_template.fillna("", inplace=True)

        overview_template = overview_template

    def get_config_aggregated_score(self, average_over_language=True, write_to_csv=False,
                                    column_name="aggregated_score"):

        # Insert aggregated mean for each model name

        self.config_aggregated_score = self.create_template()

        self.insert_config_average_scores(
            self.config_aggregated_score, average_over_language=average_over_language)

        self.config_aggregated_score = self.insert_aggregated_score_over_language_models(
            self.config_aggregated_score, column_name=column_name)

        if write_to_csv:
            if average_over_language == False:
                self.config_aggregated_score.to_csv(
                    'results/config_aggregated_scores_simple.csv')
                self.config_aggregated_score.to_excel(
                    'results/config_aggregated_scores_simple.xlsx')
            if average_over_language == True:
                self.config_aggregated_score.to_csv(
                    'results/config_aggregated_scores_average_over_language.csv')
                self.config_aggregated_score.to_excel(
                    'results/config_aggregated_scores_average_over_language.xlsx')

    def get_dataset_aggregated_score(self, average_over_language=True, write_to_csv=True, task_constraint=None):

        self.get_config_aggregated_score(
            average_over_language=average_over_language, write_to_csv=write_to_csv)

        columns = list(self.meta_infos["dataset_to_config"].keys())

        self.dataset_aggregated_score = self.create_template(columns=columns)

        for _name_or_path in self.results._name_or_path.unique():
            for dataset, configs in self.meta_infos["dataset_to_config"].items():
                if task_constraint is not None:
                    configs = [
                        conf for conf in configs if conf in task_constraint]

                dataset_mean = list()
                for conf in configs:
                    # Look if there are results for that config. If not, skip it.

                    if conf in self.config_aggregated_score.columns.tolist():
                        config_mean = self.config_aggregated_score.at[_name_or_path, conf]
                        if config_mean == "":  # This is to avoid string values; if there were no scores available I returned an empty string, because 0.0 would be missleading
                            logging.info(
                                "There is no config mean for config " + conf + " with language model " + _name_or_path)
                        elif type(config_mean) == float:
                            dataset_mean.append(config_mean)
                        else:
                            logging.error(
                                "Processed interrupted due to wrong mean value that is not a float. The mean value is: " + str(
                                    config_mean))
                            break
                    else:
                        logging.info(
                            "There seem to be no results for the configuration " + conf + " with language model " + _name_or_path)
                if len(dataset_mean) > 0:
                    if len(dataset_mean) != len(configs):
                        logging.error(
                            'Attention! It seems for dataset ' + dataset + ' you do not have the average values for configs. The average score will be calculated on the basis of incomplete information.')
                    dataset_mean = self.get_mean_from_list_of_values(
                        dataset_mean)
                else:
                    dataset_mean = ''
                self.dataset_aggregated_score.at[_name_or_path, dataset] = dataset_mean

        self.dataset_aggregated_score = self.insert_aggregated_score_over_language_models(
            self.dataset_aggregated_score)

        for finetuning_task, abbreviation in self.meta_infos["dataset_abbreviations"].items():
            try:
                self.dataset_aggregated_score.rename(
                    columns={finetuning_task: abbreviation}, inplace=True)
            except:
                pass

        if write_to_csv:
            if average_over_language == False:
                self.dataset_aggregated_score.to_csv(
                    'results/dataset_aggregated_scores_simple.csv')
                self.dataset_aggregated_score.to_excel(
                    'results/dataset_aggregated_scores_simple.xlsx')
            if average_over_language == True:
                self.dataset_aggregated_score.to_csv(
                    'results/dataset_aggregated_scores_average_over_language.csv')
                self.dataset_aggregated_score.to_excel(
                    'results/dataset_aggregated_scores_average_over_language.xlsx')

    def get_aggregated_score_for_language(self, score_type, task_constraint=None):

        tasks_relevant_for_language = list(
            self.results[(self.results[score_type].isnull() == False) & (
                    self.results[score_type] != "")].finetuning_task.unique())

        if task_constraint is not None:
            tasks_relevant_for_language = [
                t for t in tasks_relevant_for_language if t in task_constraint]

        languge_models_relevant_for_language = list(
            self.results[(self.results[score_type].isnull() == False) & (self.results[score_type] != "")][
                "_name_or_path"].unique())

        # Collect all average scores for each config grouped by dataset
        language_model_config_score_dict = defaultdict(dict)
        for ft in tasks_relevant_for_language:
            for lm in languge_models_relevant_for_language:
                # print(ft, lm, score_type)
                result = self.get_average_score(ft, lm, score_type)
                if result not in ["", np.nan]:
                    dataset_for_finetuning_task = self.meta_infos['config_to_dataset'][ft]
                    if dataset_for_finetuning_task not in language_model_config_score_dict[lm].keys():
                        language_model_config_score_dict[lm][dataset_for_finetuning_task] = list(
                        )
                    language_model_config_score_dict[lm][dataset_for_finetuning_task].append(
                        result)

        language_model_config_score_dict = dict(
            language_model_config_score_dict)

        # Average over configs per dataset
        results_averaged_over_configs = defaultdict(dict)

        for language_model, dataset_and_config_scores in language_model_config_score_dict.items():
            for dataset, config_scores in dataset_and_config_scores.items():
                results_averaged_over_configs[language_model][dataset] = self.get_mean_from_list_of_values(
                    config_scores)

        results_averaged_over_configs = dict(results_averaged_over_configs)

        # Average over datasets within language model
        results_averaged_over_datasets = dict()

        for language_model, dataset_and_score in results_averaged_over_configs.items():
            results_averaged_over_datasets[language_model] = self.get_mean_from_list_of_values(
                list(dataset_and_score.values()))

        results_averaged_over_datasets = dict(results_averaged_over_datasets)

        return results_averaged_over_datasets

    def get_language_aggregated_score(self, write_to_csv=True, task_constraint=None):

        all_languages = set()

        for languages in self.meta_infos['task_language_mapping'].values():
            for l in languages:
                # TODO: Remove this constraint in the future
                if l not in ['tr', 'nb']:
                    all_languages.add(l)

        self.language_aggregated_score = self.create_template(all_languages)

        for score_type in self.available_predict_scores:
            language = score_type.split('_')[0]
            lookup_table = self.get_aggregated_score_for_language(
                score_type, task_constraint)
            for language_model, score in lookup_table.items():
                self.language_aggregated_score.at[language_model, language] = score

        self.language_aggregated_score = self.insert_aggregated_score_over_language_models(
            self.language_aggregated_score)

        if write_to_csv:
            self.language_aggregated_score.to_csv(
                'results/language_aggregated_scores.csv')
            self.language_aggregated_score.to_excel(
                'results/language_aggregated_scores.xlsx')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose', help='Whether to enable verbose logging',
                        action='store_true', default=False)
    parser.add_argument('-pce', '--path_to_csv_export', help='Insert the path to the exported csv file from wandb',
                        default=None)
    parser.add_argument('-pn', '--project_name', help='Insert the wandb project name',
                        default=None)
    parser.add_argument('-wl', '--which_language',
                        help='Default value is None. Choose which language should be considered for the overviews. '
                             'If set to `monolingual`, only monolingual models will be considered. '
                             'If set to `multilingual`, only multilingual models will be considered. '
                             'Otherwise, only those models will be considered that belong to the defined language. '
                             'For example, if you pass `de`, only German models will be considered.')
    parser.add_argument('-wak', '--wandb_api_key',
                        help='To be able to fetch the right results, you can insert the wand api key. '
                             'Alternatively, set the WANDB_API_KEY environment variable to your API key.',
                        default=None)

    args = parser.parse_args()

    with open('tasks_for_report.json', 'r') as f:
        tasks_for_report = js.load(f)

    ra = ResultAggregator(wandb_api_key=args.wandb_api_key,
                          project_name=args.project_name,
                          path_to_csv_export=args.path_to_csv_export,
                          verbose_logging=args.verbose,
                          only_completed_tasks=False,
                          which_language=args.which_language)

    ra.get_info()

    ra.create_report(tasks_to_filter=tasks_for_report['finetuning_task'])
    # ra.create_report()

    ra.get_dataset_aggregated_score()

    ra.get_language_aggregated_score()
