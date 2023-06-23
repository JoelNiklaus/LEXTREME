from datetime import datetime
from pathlib import Path
from statistics import mean
import argparse
from collections import defaultdict
from copy import deepcopy
from helper import insert_responsibilities
from revision_handler import RevisionHandler
from scipy.stats import hmean, gmean
from traceback import print_exc
from typing import List, Dict, AnyStr, Union, Literal, Optional, Set
import json as js
import logging
import numpy as np
import os
import pandas as pd
import re
import sys
import wandb
from pprint import pprint
from helper import *

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utilities import get_meta_infos

meta_infos = get_meta_infos()


class ResultAggregator(RevisionHandler):
    meta_infos = meta_infos

    def __init__(self,
                 report_spec_name: str,
                 only_completed_tasks: bool = False,
                 path_to_csv_export: str = None,
                 project_name: str = None,
                 output_dir: str = TABLE_OUTPUT_PATH,
                 log_dir: str = 'logs',
                 score: str = 'macro-f1',
                 split: str = "predict",
                 verbose_logging: bool = True,
                 fill_with_wrong_revisions: bool = False,
                 wandb_api_key: Optional[str] = None,
                 which_language: Optional[str] = None,
                 required_seeds: Optional[List[int]] = None,
                 mean_type='harmonic',
                 remove_outliers: bool = False
                 ):
        super().__init__(report_spec_name)
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
        fill_with_wrong_revisions: Default is False. If set to True, the code will check which runs are missing for which seed and try to fill these missing seeds with runs that actually do not fullfil, because the wrong revision was used.
        """
        # Create the result directory if not existent
        self.output_dir = os.path.join(output_dir, f'results_based_on_{mean_type}_mean')
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        if split == "eval":
            self.split = split + "/"
        elif split == "predict":
            self.split = split + "/_"

        self.score = score

        self.api = wandb.Api()

        name_of_log_file = f'{self.log_dir}/logs_' + \
                           datetime.now().isoformat() + '.log'
        name_of_log_file = os.path.join(
            os.path.dirname(__file__), name_of_log_file)

        self.only_completed_tasks = only_completed_tasks

        self.which_language = which_language

        if verbose_logging == True:
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

        if required_seeds is None:
            self.required_seeds = [1, 2, 3]
        else:
            self.required_seeds = required_seeds

        self.fill_with_wrong_revisions = fill_with_wrong_revisions

        if type(wandb_api_key) == str and bool(re.search(r'\d', wandb_api_key)):
            os.environ["WANDB_API_KEY"] = self.wandb_api_key

        self._path = name_of_log_file
        if path_to_csv_export is None:
            results = self.get_wandb_overview(project_name)
            results.to_csv(
                f"{self.output_dir}/current_wandb_results_unprocessed.csv", index=False)
        else:
            results = pd.read_csv(Path(path_to_csv_export))
            results = self.edit_result_dataframe(results, name_editing=False)
        results = results[results.finetuning_task.isnull() == False]

        self.results = results

        available_predict_scores = [col for col in self.results if bool(re.search(r'^\w+_predict/_' + self.score, col))]
        self.available_predict_scores = sorted(available_predict_scores)
        self.available_predict_scores_original = deepcopy(self.available_predict_scores)

        self.mean_type = mean_type

        self.remove_outliers = remove_outliers

        self.removed_values_dict = defaultdict(dict)

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

    def remove_slashes(self, string: str) -> str:

        """
        The function will remove any slash in a string. This is necessary sometimes, e.g. in excel sheet names must
        not contain slashes.
        @param string: String value that might contain a slash.
        @return: String value without slash.
        """

        return re.sub(r'/', '', string)

    def generate_concrete_score_name(self) -> Set[str]:

        """
        The function will generate the names of all necessary metrics that are logged to wandb and that might be used
        for analyses.
        """
        # TODO make constant out of this
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

        score_names.add('eval/loss')
        score_names.add('predict/_loss')

        return score_names

    def fetch_data(self, project_name: str) -> List[dict]:

        """
        Fetches the data from the wandb project and saves them to a list of dictionaries.

        @param project_name: name of the wandb project
        @return: List of dictionaries with the necessary logs from the wandb project
        """

        # Project is specified by <entity/project-name>
        runs = self.api.runs(project_name)

        score_names = self.generate_concrete_score_name()

        results = list()
        for x in runs:
            entry = dict()
            try:
                if x.state == "finished":
                    already_added_fields = set()
                    entry['project_name'] = project_name
                    entry['state'] = x.state
                    if x.config["finetuning_task"] == 'case_hold':
                        entry["finetuning_task"] = 'en_' + \
                                                   x.config["finetuning_task"]
                    else:
                        entry["finetuning_task"] = x.config['finetuning_task']

                    already_added_fields.add('finetuning_task')

                    if '_name_or_path' in x.config.keys():
                        entry["_name_or_path"] = x.config['_name_or_path']
                        already_added_fields.add('_name_or_path')
                    else:
                        entry["_name_or_path"] = x.summary['model_name_or_path']

                    for field, value in x.config.items():
                        if field not in already_added_fields:
                            entry[field] = value
                    # entry["seed"] = x.config['seed']
                    entry['name'] = x.name
                    if 'revision' in x.summary.keys():
                        entry['revision'] = x.summary.revision
                    else:
                        entry['revision'] = 'main'
                    for sn in score_names:
                        if sn in x.history_keys['keys'].keys():
                            entry[sn] = x.history_keys['keys'][sn]['previousValue']
                        else:
                            entry[sn] = ''
                    results.append(entry)
            except:
                print_exc()

        return results

    def get_wandb_overview(self, project_name: str) -> pd.core.frame.DataFrame:
        """
        Covers several other functions that fetch the data from wandb and preprocess it.
        @param project_name: Name of the wandb project
        @return: pd.core.frame.DataFrame
        """
        # if project_name is None:
        # project_name = "lextreme/paper_results"
        # results = wandb_summarizer.download.get_results(project_name=project_name)
        # results = pd.DataFrame(self.fetch_data(project_name=project_name))
        # results_old_lex_glue_runs = pd.DataFrame(self.fetch_data(project_name='old_lex_glue_runs'))
        # results = pd.concat([results, results_old_lex_glue_runs])
        results = self.fetch_data(project_name=project_name)
        results = pd.DataFrame(results)
        results.to_csv(f"{self.output_dir}/current_wandb_results_unprocessed.csv", index=False)
        results = self.edit_result_dataframe(results)
        if self.which_language is not None:
            results = self.filter_by_language(results, self.which_language)
        return results

    def edit_column_name(self, column_name: str) -> str:
        if column_name.startswith('config_'):
            column_name = re.sub('^config_', '', column_name)
        if column_name.startswith('end'):
            column_name = re.sub('^end_', '', column_name)
        return column_name

    def insert_hierarchical(self, run_name: str) -> bool:
        """
        Infos whether the run was hierarchical were not logged to wandb in the beginning of the project.
        But this information is contained in the name of the run.
        This function will extract this information.
        Newer runs do contain a field named hierarchical.
        """

        if 'hierarchical_True' in run_name:
            return True
        elif 'hierarchical_False' in run_name:
            return False

    def edit_column_names_in_df(self, dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        columns = dataframe.columns.tolist()

        for col in columns:
            col_new = self.edit_column_name(col)
            dataframe = dataframe.rename(columns={col: col_new})

        return dataframe

    def update_language_specifc_predict_columns(self, dataframe: pd.core.frame.DataFrame,
                                                score=None) -> pd.core.frame.DataFrame:
        """
        For multilingual datasets we insert the overall macro-f1 score into the column for the language-specific
        macro-f1 score, if a monolingual model was fine-tuned on them.
        Similarly, for monolingual datasets we insert the overall macro-f1 into the field for the language-specific
        macro-f1 score.
        For example, if a German model was fine-tuned on mapa_fine, we filter mapa_fine by German examples.
        Therefore, the value for predict/_micro-f1 holds true for de_predict/_micro-f1 as well.
        """

        if score is None:
            score = self.split + self.score

        monolingual_configs = []
        multilingual_configs = []

        for config, languages in self.meta_infos['task_language_mapping'].items():
            if len(languages) == 1:
                monolingual_configs.append(config)
            else:
                multilingual_configs.append(config)

        # all_available_languages_for_training_set = list(dataframe['language'].unique())
        # all_available_languages_for_training_set = [l for l in all_available_languages_for_training_set if l != "all"]

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
                    if not isinstance(dataframe.at[i, column_name], float):
                        dataframe.at[i, column_name] = dataframe.at[i, score]

        return dataframe

    def language_and_model_match(self, _name_or_path: str, finetuning_task: str, language: str) -> bool:

        """
        Caution: This function was introduced, because in the beginning we wanted to fine-tune multilingual models
        on monolingual subsets of multilingual datasets. This meant, that the language of the model and the language of
        the dataset would not match, because we would have a multilingual model (with the language 'all')
        and, for example, a German subset of a multilingual dataset with the language 'de'. But later, we decided that
        we would not do that which is why we need to filter out these cases.

        @param language_model: Name of the language model
        @finetuning_task: Name of the finetuning task
        @param language: Value of the field 'language'
        @return: bool
        """
        if len(self.meta_infos["task_language_mapping"][finetuning_task]) > 1:
            return self.meta_infos["model_language_lookup_table"][_name_or_path] == language
        else:
            return True

    def correct_language(self, finetuning_task, language_model, language):
        """
        Sometimes the language column is wrong.
        For example, for greek_legal_code_subject_level in combination with distilbert-base-multilingual-cased,
        the run names start with “all_” instead of “el_” which should not be.
        This code will check if a language model is multilingual and if a task is monolingual.
        If the language is all, it must be corrected to the language of the monolingual model.
        """
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

    def check_for_model_language_consistency(self, dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        dataframe['language'] = dataframe.apply(
            lambda x: self.correct_language(x["finetuning_task"], x["_name_or_path"], x["language"]), axis=1)

        dataframe['model_language_consistency'] = dataframe.apply(
            lambda x: self.language_and_model_match(_name_or_path=x["_name_or_path"],
                                                    finetuning_task=x["finetuning_task"],
                                                    language=x["language"]), axis=1)

        # Remove this part if you want to consider multilingual models trained on monolingual subsets of multilingual models
        dataframe = dataframe[dataframe.model_language_consistency == True]
        del dataframe["model_language_consistency"]

        return dataframe

    def loss_equals_nan(self, loss: float) -> bool:

        """
        For some reason, when wandb shows loss == nan, you will not get a nan value from the API.
        Instead, it returns a very low value in scientific notation.
        This functions tries to detect these cases.
        A manuel evaluation showed that the predictions are worse in these cases, so we need to rerun the experiments again.

        """

        return '+' in str(loss)

    def find_best_revisions(self, dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

        """
        @param dataframe: pandas dataframe that contains the fetched wandb data.
        @return: pandas dataframe
        """
        dataframe = dataframe.sort_values(['finetuning_task', '_name_or_path', 'seed', 'revision'])
        dataframe_new = list()
        available_finetuning_tasks = dataframe.finetuning_task.unique()
        for finetuning_task in available_finetuning_tasks:
            for _name_or_path, revision in self.revision_lookup_table.items():
                sub_dataframe = dataframe[
                    (dataframe.finetuning_task == finetuning_task) & (dataframe._name_or_path == _name_or_path)]
                for row in sub_dataframe.to_dict(orient="records"):
                    if row['_name_or_path'] == _name_or_path and row['finetuning_task'] == finetuning_task:
                        if self.revision_does_match(_name_or_path=row['_name_or_path'],
                                                    revision_from_wandb=row['revision']):
                            dataframe_new.append(row)
                seeds_extracted = {row['seed'] for row in dataframe_new if
                                   row['finetuning_task'] == finetuning_task and row['_name_or_path'] == _name_or_path}
                if seeds_extracted != set(self.required_seeds):
                    missing_seeds = [s for s in self.required_seeds if s not in seeds_extracted]
                    print('Missing seeds: ', missing_seeds)
                    for seed in missing_seeds:
                        for row in sub_dataframe.to_dict(orient="records"):
                            if row['_name_or_path'] == _name_or_path and row['finetuning_task'] == finetuning_task and \
                                    row['seed'] == seed:
                                dataframe_new.append(row)

        dataframe_new = pd.DataFrame(dataframe_new)

        return dataframe_new

    def edit_result_dataframe(self, results: pd.core.frame.DataFrame,
                              name_editing: bool = True) -> pd.core.frame.DataFrame:

        """
        @param results (pd.core.frame.DataFrame): Fetched results from wandb in form of a pandas dataframe.
        @param name_editing (bool): If set to `True`, some columns will be renamed.
        @return: pd.core.frame.DataFrame
        """
        results = self.edit_column_names_in_df(results)

        # Filter out any model that is not listed in meta infos
        results = results[results._name_or_path.isin(meta_infos["all_available_models"])]

        # all_finetuning_tasks = results.finetuning_task
        if name_editing == True:
            results['language'] = results.finetuning_task.apply(lambda x: str(x).split('_')[0])
            results['finetuning_task'] = results.finetuning_task.apply(lambda x: '_'.join(str(x).split('_')[1:]))
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

        # Remove all cases where eval/loss is nan according to wandb
        results = results[results['predict/_loss']
                          .apply(self.loss_equals_nan) == False]

        # Keep only results from seed 1,2,3 and remove anything else
        results = results[results.seed.isin(self.required_seeds)]

        # When fetching the data from the wandb api it returns state instead of State as column name
        # I could make every column name lowercase, but I am afraid this might cause some problems
        if 'State' in results.columns:
            results = results[results.State == "finished"]
        elif 'state' in results.columns:
            results = results[results.state == "finished"]

        # Check if revisions match
        results['revisions_match'] = results.apply(
            lambda row: self.revision_does_match(_name_or_path=row['_name_or_path'],
                                                 revision_from_wandb=row['revision']), axis=1)
        if self.fill_with_wrong_revisions == True:
            results = self.find_best_revisions(results)
        else:
            results = results[results.revisions_match == True]

        results = results.sort_values(["finetuning_task", "_name_or_path", "seed", "predict/_" + self.score])
        results = results.drop_duplicates(["seed", "finetuning_task", "_name_or_path", "language"], keep='last')

        return results

    def remove_languages(self, languages: Union[list, str]) -> None:

        """
        Will remove a language from the list of available languages for the tables that will be generated.
        THis might be useful, if we want to exclude some languages from the overview.

        @param languages: Language ISO 639-1 code.
        @return: None

        """
        if isinstance(languages, str):
            self.available_predict_scores = [score for score in self.available_predict_scores if
                                             score.startswith(languages) == False]
        elif isinstance(languages, list):
            for lang in languages:
                self.available_predict_scores = [score for score in self.available_predict_scores if
                                                 score.startswith(lang) == False]

    def reset_list_of_available_languages(self) -> None:

        """
        If in a previous step the list of available languages was altered, this function will reset the list of
        available languages to the initial state.
        @return: None
        """
        self.available_predict_scores = deepcopy(self.available_predict_scores_original)

    def filter_by_language(self, dataframe: pd.core.frame.DataFrame, which_language: str) -> pd.core.frame.DataFrame:

        """
        Sometimes, we are interested in the results of a specific language. This function will filter the data by
        language.

        @param dataframe: Data fetched from the wandb project.
        @param which_language: Either `monolingual`, `multilingual` or ISO 639-1.
        @return: pd.core.frame.DataFrame
        """

        if which_language == MONOLINGUAL:
            dataframe = dataframe[
                dataframe._name_or_path.apply(lambda x: meta_infos['model_language_lookup_table'][x] != 'all')]
        elif which_language == MULTILINGUAL:
            dataframe = dataframe[
                dataframe._name_or_path.apply(lambda x: meta_infos['model_language_lookup_table'][x] == 'all')]
        else:
            dataframe = dataframe[
                dataframe._name_or_path.apply(
                    lambda x: meta_infos['model_language_lookup_table'][x] == which_language)]
        return dataframe

    def get_available_models_per_task(self, finetuning_task: str) -> list:

        required_models = list()
        languages = self.meta_infos["task_language_mapping"][finetuning_task]
        for model_name, language in self.meta_infos['model_language_lookup_table'].items():
            if language == "all":
                required_models.append(model_name)
            else:
                for lang in languages:
                    if lang == language:
                        required_models.append(model_name)

        available_models = self.results[(self.results.finetuning_task == finetuning_task)
                                        & (self.results._name_or_path.isin(required_models))]._name_or_path.unique()

        return required_models, available_models

    def check_seed_per_task(self, task_constraint: list = [], model_constraint: list = [], which_language=None,
                            required_seeds=None) -> pd.core.frame.DataFrame:

        """
        Will create a dataframe of incomplete finetuning tasks that contains the name of the fnetuning task, the model name,
        the required revision and the missing seeds.

        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @param which_language (optional) : Either `monolingual`, `multilingual` or ISO 639-1 to filter by language.
        @param required_seeds (optional): List of seed that want to consider. If None, the default value will be taken.
        @return: pd.core.frame.DataFrame
        """
        if required_seeds is None:
            required_seeds = self.required_seeds
        if which_language is None:
            which_language = self.which_language

        report = list()

        for finetuning_task in self.meta_infos["task_language_mapping"].keys():

            required_models, available_models = self.get_available_models_per_task(finetuning_task=finetuning_task)

            for required_name_or_path in required_models:
                if required_name_or_path not in available_models:
                    message = "For task " + finetuning_task + " in combination with the language " + \
                              self.meta_infos['model_language_lookup_table'][
                                  required_name_or_path] + " we do not have any results for this model: " + required_name_or_path
                    logging.warning(message)
                    item = dict()
                    item['finetuning_task'] = finetuning_task
                    item['_name_or_path'] = required_name_or_path
                    item['language'] = self.meta_infos['model_language_lookup_table'][required_name_or_path]
                    item['missing_seeds'] = [
                        str(s) for s in sorted(list(required_seeds))]
                    report.append(item)

            for available_name_or_path in available_models:
                list_of_seeds = set(self.results[(self.results.finetuning_task == finetuning_task) & (
                        self.results._name_or_path == available_name_or_path)].seed.unique())
                list_of_seeds = set([str(int(x)) for x in list_of_seeds])
                if list_of_seeds.intersection(required_seeds) != required_seeds:
                    missing_seeds = set([str(s) for s in required_seeds if s not in list_of_seeds])
                    message = "There not enough seeds for task " + finetuning_task + " in combination with the language model " \
                              + available_name_or_path + ". We have only results for the following seeds: ", ', '.join(
                        list(list_of_seeds)) + ". Results are missing for the following seeds: ", ', '.join(
                        list(missing_seeds))
                    logging.warning(message)
                    item = dict()
                    item['finetuning_task'] = finetuning_task
                    item['_name_or_path'] = available_name_or_path
                    item['language'] = self.meta_infos['model_language_lookup_table'][available_name_or_path]
                    item['missing_seeds'] = sorted(list(missing_seeds))
                    report.append(item)

        report_df = pd.DataFrame(report)
        report_df['missing_seeds'] = report_df.missing_seeds.apply(lambda x: ','.join(x))

        report_df = report_df.drop_duplicates()

        if which_language is not None:
            report_df = self.filter_by_language(report_df, which_language)

        report_df['revision'] = report_df.apply(
            lambda x: self.insert_revision(x['_name_or_path']), axis=1)

        report_df = insert_responsibilities(report_df)

        if len(task_constraint) > 0:
            report_df = report_df[report_df.finetuning_task.isin(task_constraint)]

        if len(model_constraint) > 0:
            report_df = report_df[report_df._name_or_path.isin(model_constraint)]

        return report_df

    def get_list_of_incomplete_tasks(self, which_language: str = None,
                                     task_constraint: list = [],
                                     model_constraint: list = [],
                                     required_seeds: list = None) -> list:

        """
        Returns a list of finetuning tasks that still have missing seeds.
        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @param which_language (optional) : Either `monolingual`, `multilingual` or ISO 639-1 to filter by language.
        @param required_seeds (optional): List of seed that want to consider. If None, the default value will be taken.
        @return: List[int]
        """

        seed_check = self.check_seed_per_task(task_constraint=task_constraint, model_constraint=model_constraint,
                                              required_seeds=required_seeds, which_language=which_language)

        incomplete_tasks = set(seed_check.finetuning_task.unique())

        return incomplete_tasks

    def create_overview_of_results_per_seed(self, score: str = None,
                                            only_completed_tasks: bool = None,
                                            which_language: str = None,
                                            task_constraint: list = [],
                                            model_constraint: list = [],
                                            required_seeds: list = None) -> pd.core.frame.DataFrame:

        """
        Returns a dataframe with the score for each seed for each fine-tuning task with each model.

        @param score: str: Name of the score you wish to be displayed. Available scores can be found under self.available_predict_scores_original
        @param only_completed_tasks: boolean: Define whether you want to consider only complete tasks or not.
        @param which_language (optional) : Either `monolingual`, `multilingual` or ISO 639-1 to filter by language.
        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @param required_seeds (optional): List of seed that want to consider. If None, the default value will be taken.
        @return: -> pd.core.frame.DataFrame
        """
        if required_seeds is None:
            required_seeds = self.required_seeds
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

        required_seeds = list(required_seeds)
        required_seeds = sorted(int(s) for s in required_seeds)

        # If score contains the name of the split in the name, no further postprocessing is required.
        if bool(re.search('(eval|train)', score)):
            score_to_filter = score
        else:
            score_to_filter = self.split + score

        if only_completed_tasks == True:
            incomplete_tasks = self.get_list_of_incomplete_tasks(task_constraint=task_constraint,
                                                                 model_constraint=model_constraint,
                                                                 required_seeds=required_seeds,
                                                                 which_language=which_language)
            # Caution! If there is no completed task, it can happen that the resulting dataframe is empty!
            df = self.results[(self.results.seed.isin(required_seeds)) &
                              (self.results.finetuning_task.isin(incomplete_tasks) == False)][['finetuning_task',
                                                                                               '_name_or_path',
                                                                                               'language',
                                                                                               'seed',
                                                                                               score_to_filter]]

        elif only_completed_tasks == False:
            df = self.results[(self.results.seed.isin(required_seeds))][
                ['finetuning_task', '_name_or_path', 'language', 'seed', score_to_filter]]

        if len(task_constraint) > 0:
            df = df[df.finetuning_task.isin(task_constraint)]

        if len(model_constraint) > 0:
            df = df[df._name_or_path.isin(model_constraint)]

        df_pivot = df.pivot_table(values=score_to_filter,
                                  index=['finetuning_task', '_name_or_path', 'language'],
                                  columns='seed',
                                  aggfunc='first')

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
            for seed in required_seeds:
                if seed not in df_pivot_columns:
                    df_pivot[seed] = ''

        relevant_columns = ['dataset', 'finetuning_task',
                            'task_type', '_name_or_path', 'language']
        relevant_columns.extend(required_seeds)
        df_pivot = df_pivot[relevant_columns]
        df_pivot['arithmetic_mean_over_seeds'] = df_pivot.mean(axis=1, numeric_only=True)

        if old_score:
            self.score = old_score

        # Add column with best seed
        # df_pivot.fillna('', inplace=True)
        for i, _ in df_pivot.iterrows():
            all_seed_values = list()
            for seed in required_seeds:
                seed_value = df_pivot.at[i, seed]
                all_seed_values.append((seed, seed_value))

            if len(all_seed_values) > 0:
                all_seed_values_sorted = sorted(
                    all_seed_values, key=lambda x: x[1], reverse=True)
                best_seed = all_seed_values_sorted[0][0]
                best_seed_value = all_seed_values_sorted[0][1]
            else:
                best_seed = ''
                best_seed_value = ''
            df_pivot.at[i, 'best_seed'] = best_seed
            df_pivot.at[i, 'best_seed_value'] = best_seed_value
            df_pivot['standard_deviation'] = df_pivot[required_seeds].std(
                axis=1)

        return df_pivot

    def create_report(self, only_completed_tasks: bool = None,
                      which_language: str = None,
                      task_constraint: list = [],
                      model_constraint: list = [],
                      required_seeds: list = None) -> pd.core.frame.DataFrame:

        """
        Will create several overviews of different scores, such as macor-f1, micro-f1 etc. and save them in an excel file.

        @param only_completed_tasks: boolean: Define whether you want to consider only complete tasks or not.
        @param which_language (optional) : Either `monolingual`, `multilingual` or ISO 639-1 to filter by language.
        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @param required_seeds (optional): List of seed that want to consider. If None, the default value will be taken.
        @return: -> pd.core.frame.DataFrame
        """

        if required_seeds is None:
            required_seeds = self.required_seeds

        required_seeds = {str(s) for s in required_seeds}

        if only_completed_tasks is None:
            only_completed_tasks = self.only_completed_tasks

        if which_language is None:
            which_language = self.which_language

        seed_check = self.check_seed_per_task(task_constraint=task_constraint, model_constraint=model_constraint,
                                              required_seeds=required_seeds, which_language=which_language)
        self.seed_check = seed_check

        macro_f1_overview = self.create_overview_of_results_per_seed(score="macro-f1",
                                                                     only_completed_tasks=only_completed_tasks,
                                                                     which_language=which_language,
                                                                     task_constraint=task_constraint,
                                                                     model_constraint=model_constraint,
                                                                     required_seeds=required_seeds
                                                                     )
        self.macro_f1_overview = macro_f1_overview
        micro_f1_overview = self.create_overview_of_results_per_seed(score="micro-f1",
                                                                     only_completed_tasks=only_completed_tasks,
                                                                     which_language=which_language,
                                                                     task_constraint=task_constraint,
                                                                     model_constraint=model_constraint,
                                                                     required_seeds=required_seeds
                                                                     )
        self.micro_f1_overview = micro_f1_overview
        weighted_f1_overview = self.create_overview_of_results_per_seed(score="weighted-f1",
                                                                        only_completed_tasks=only_completed_tasks,
                                                                        which_language=which_language,
                                                                        task_constraint=task_constraint,
                                                                        model_constraint=model_constraint,
                                                                        required_seeds=required_seeds
                                                                        )
        self.weighted_f1_overview = weighted_f1_overview
        accuracy_normalized_overview = self.create_overview_of_results_per_seed(score="accuracy_normalized",
                                                                                only_completed_tasks=only_completed_tasks,
                                                                                which_language=which_language,
                                                                                task_constraint=task_constraint,
                                                                                model_constraint=model_constraint,
                                                                                required_seeds=required_seeds
                                                                                )
        self.accuracy_normalized_overview = accuracy_normalized_overview

        eval_macro_f1_overview = self.create_overview_of_results_per_seed(
            score="eval/macro-f1", only_completed_tasks=only_completed_tasks, which_language=which_language,
            task_constraint=task_constraint, model_constraint=model_constraint, required_seeds=required_seeds)

        self.eval_macro_f1_overview = eval_macro_f1_overview

        eval_micro_f1_overview = self.create_overview_of_results_per_seed(
            score="eval/micro-f1", only_completed_tasks=only_completed_tasks, which_language=which_language,
            task_constraint=task_constraint, model_constraint=model_constraint, required_seeds=required_seeds)

        self.eval_micro_f1_overview = eval_micro_f1_overview

        # write to csv, so we can read it with a bash script
        self.seed_check.to_csv(
            f"{self.output_dir}/completeness_report.tsv", sep="\t", index=False)

        with pd.ExcelWriter(f'{self.output_dir}/report.xlsx') as writer:
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
            self.eval_micro_f1_overview.to_excel(
                writer, index=False, sheet_name="eval_micro__f1_overview")

    def postprocess_columns(self, dataframe: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

        """
        Applies some postprocessing of the results in a dataframe before exporting the dataframe to a latex table.
        @param dataframe: pd.core.frame.DataFrame.
        @return: pd.core.frame.DataFrame
        """
        dataframe.fillna('', inplace=True)
        empty_columns = dataframe.columns[dataframe.eq('').all()]
        dataframe = dataframe.drop(empty_columns, axis=1)
        dataframe = dataframe.applymap(lambda x: round_value(x))
        dataframe = dataframe.applymap(lambda x: replace_empty_string(x, np.nan))
        if self.model_order_acc_to_report_specs and len(self.model_order_acc_to_report_specs) == dataframe.shape[0]:
            dataframe = dataframe.reindex(self.model_order_acc_to_report_specs)

        # Insert Model name abbreviations
        if set(dataframe.index.tolist()) == set(self.model_order_acc_to_report_specs):
            dataframe = dataframe.rename(index=self.meta_infos["model_abbrevations"])

        return dataframe

    def convert_numpy_float_to_python_float(self, value) -> Union[int, str]:

        """
        @param value: Any value.
        @return:
        """
        if value == np.nan:
            final_value = ""
        elif isinstance(value, np.floating):
            final_value = value.item()
        else:
            final_value = value

        return final_value

    def get_mean_from_list_of_values(self, list_of_values: List[float],
                                     with_standard_deviation: bool = False) -> Union[tuple, int, str]:

        """
        Returns the mean from a list of floats.

        @param list_of_values: List with floats.
        @param with_standard_deviation: Boolean value. If set to True, it will return the standard deviations.
        @return:
        """

        if self.mean_type == 'harmonic':
            mean_function = hmean
        elif self.mean_type == 'geometric':
            mean_function = gmean
        elif self.mean_type is None or self.mean_type == 'arithmetic':
            mean_function = mean
        # There used to be this condition and x > 0, but sometimes there are really scores of 0
        list_of_values = [x for x in list_of_values if isinstance(x, float) and not pd.isnull(x)]

        if len(list_of_values) > 0 and with_standard_deviation == False:
            return self.convert_numpy_float_to_python_float(mean_function(list_of_values))
        elif len(list_of_values) > 0 and with_standard_deviation == True:
            return (mean_function(list_of_values), np.std(list_of_values))
        else:
            return ""

    def write_to_files(self, dataframe_with_aggregate_scores: pd.core.frame.DataFrame,
                       filename: str,
                       average_over_language: bool = True,
                       with_standard_deviation: bool = False) -> None:

        if average_over_language == False:
            dataframe_with_aggregate_scores.to_csv(
                f'{self.output_dir}/{filename}_simple.csv')
            if with_standard_deviation == False:
                dataframe_with_aggregate_scores.style.highlight_max(color='lightgreen', axis=0).to_excel(
                    f'{self.output_dir}/{filename}_simple.xlsx')
            if with_standard_deviation == True:
                dataframe_with_aggregate_scores.to_excel(
                    f'{self.output_dir}/{filename}_simple.xlsx')
            make_latext_table(dataframe_with_aggregate_scores, f'{self.output_dir}/{filename}_simple.tex')
        if average_over_language == True:
            dataframe_with_aggregate_scores.to_csv(
                f'{self.output_dir}/{filename}_average_over_language.csv')
            if with_standard_deviation == False:
                dataframe_with_aggregate_scores.style.highlight_max(color='lightgreen', axis=0).to_excel(
                    f'{self.output_dir}/{filename}_average_over_language.xlsx')
            if with_standard_deviation == True:
                dataframe_with_aggregate_scores.to_excel(f'{self.output_dir}/{filename}_average_over_language.xlsx')
            make_latext_table(dataframe_with_aggregate_scores,
                              f'{self.output_dir}/{filename}_average_over_language.tex')

    def get_average_score(self, finetuning_task: str, _name_or_path: str, score: str = None,
                          with_standard_deviation: bool = False) -> float:
        if score is None:
            score = 'predict/_' + self.score

        results_filtered = self.results[(self.results.finetuning_task == finetuning_task)
                                        & (self.results._name_or_path == _name_or_path)][["seed", score]]

        if len(results_filtered.seed.unique()) < 3:
            logging.warning(
                "Attention. For the fine-tuning task " + finetuning_task + " with model name " + _name_or_path + " you have " + str(
                    len(results_filtered.seed.unique())) + " instead of 3 seeds! The mean will be calculated anyway.")

        if len(results_filtered.seed.unique()) != len(results_filtered.seed.tolist()):
            logging.warning(
                'Attention! It seems you have duplicate seeds for the fine-tuning task ' + finetuning_task + " with the model " + _name_or_path)
            logging.info(
                'Duplicate values will be removed based on the seed number.')

        if len(results_filtered.seed.tolist()) == 0:
            logging.info('The fine-tuning task ' + finetuning_task +
                         ' with the model ' + _name_or_path + " has no meaningful results.")
            return ""  # There is nothing to be calculated
        else:
            if self.remove_outliers == True:
                list_of_values, removed_values = remove_outliers_in_values(
                    list_of_values=results_filtered[score].tolist())
                if len(removed_values) > 0:
                    self.removed_values_dict[finetuning_task][_name_or_path] = removed_values
            elif self.remove_outliers == False:
                list_of_values = results_filtered[score].tolist()

            mean_value = self.get_mean_from_list_of_values(list_of_values,
                                                           with_standard_deviation=with_standard_deviation)

        if mean_value in ["", np.nan]:
            logging.warning(
                f'There is an error for the fine-tuning task {finetuning_task} with the model {_name_or_path} and the score {score}. '
                f'There are no useful values to calculate an mean score. ')
        elif isinstance(mean_value, str):
            logging.warning(
                f'There is an error for the fine-tuning task {finetuning_task} with the model {_name_or_path} and the score {score}. '
                f'There are no useful values to calculate an mean score. ')
        return mean_value

    def create_template(self, columns: list = None) -> pd.core.frame.DataFrame:

        """
        Creates an empty dataframe with predefined indices and column names that will be used to create the dataframe
        with the aggregate scores.

        @param columns: List of column names that you want to add.
        @return: pd.core.frame.DataFrame
        """

        # Create empty dataframe
        overview_template = pd.DataFrame()

        # Se the list of model names as index
        overview_template = overview_template.set_axis(self.results._name_or_path.unique())

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

    def get_config_aggregated_score(self, average_over_language=True, write_to_csv=False,
                                    column_name="Agg.", task_constraint: list = [],
                                    model_constraint: list = [], with_standard_deviation=False):

        """
        Creates the dataframe with the config aggregate scores.

        @param average_over_language: Boolean Value. If set to 'True', we will average over languages. Default is True.
        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @param with_standard_deviation: Boolean value. Default is False. Whether you want to add standard deviation or not.
        @param column_name: Name of the column that presents the aggregate score.
        @return: None.
        """

        columns = list()
        for _, configs in self.meta_infos["dataset_to_config"].items():
            if len(task_constraint) > 0:
                configs = [c for c in configs if c in task_constraint]
            if len(configs) > 0:
                columns.extend(configs)

        self.config_aggregated_score = self.create_template(columns=columns)

        self.insert_config_average_scores(overview_template=self.config_aggregated_score,
                                          average_over_language=average_over_language,
                                          task_constraint=task_constraint,
                                          model_constraint=model_constraint,
                                          with_standard_deviation=with_standard_deviation)

        self.config_aggregated_score = self.insert_aggregated_score_over_language_models(
            self.config_aggregated_score, column_name=column_name)

        if len(model_constraint) > 0:
            self.config_aggregated_score = self.config_aggregated_score[
                self.config_aggregated_score.index.isin(model_constraint)]

        config_aggregated_score = deepcopy(self.config_aggregated_score)
        config_aggregated_score = insert_abbreviations_in_columns(config_aggregated_score)
        config_aggregated_score = self.postprocess_columns(config_aggregated_score)
        if write_to_csv:
            self.write_to_files(dataframe_with_aggregate_scores=config_aggregated_score,
                                filename='config_aggregated_scores',
                                average_over_language=average_over_language,
                                with_standard_deviation=with_standard_deviation)

    def insert_config_average_scores(self, overview_template: pd.core.frame.DataFrame,
                                     average_over_language: bool = True,
                                     task_constraint: list = [],
                                     model_constraint: list = [],
                                     with_standard_deviation: bool = False) -> None:
        """

        @param overview_template: Template for the dataframe that contains the scores.
        @param average_over_language: Boolean Value. If set to 'True', we will average over languages. Default is True.
        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @param with_standard_deviation: Boolean value. Default is False. Whether you want to add standard deviation or not.
        @return: None.
        """
        logging.info("*** Calculating average scores ***")
        allowed_models = list(self.results._name_or_path.unique())
        allowed_tasks = list(self.results.finetuning_task.unique())

        if len(model_constraint) > 0:
            allowed_models = [
                am for am in allowed_models if am in model_constraint]

        if len(task_constraint) > 0:
            allowed_tasks = [
                at for at in allowed_tasks if at in task_constraint]

        if average_over_language == False:
            for _name_or_path in allowed_models:
                for finetuning_task in allowed_tasks:
                    mean_macro_f1_score = self.get_average_score(finetuning_task=finetuning_task,
                                                                 _name_or_path=_name_or_path,
                                                                 with_standard_deviation=with_standard_deviation)
                    overview_template.at[_name_or_path, finetuning_task] = mean_macro_f1_score

        elif average_over_language == True:
            for _name_or_path in allowed_models:
                for finetuning_task in allowed_tasks:

                    # We check of the fine-tuning task has more than one language
                    # If not, we can process with the normal predict/_macro-f1 that stands for the entire config
                    # If yes, we loop through all available language-specific macro-f1 scores

                    if len(self.meta_infos["task_language_mapping"][finetuning_task]) == 1:
                        mean_macro_f1_score = self.get_average_score(finetuning_task=finetuning_task,
                                                                     _name_or_path=_name_or_path,
                                                                     with_standard_deviation=with_standard_deviation)

                    elif len(self.meta_infos["task_language_mapping"][finetuning_task]) > 1:
                        predict_language_mean_collected = list()
                        predict_language_collected = list()

                        # Average over languages
                        for aps in self.available_predict_scores:
                            language = aps.split('_')[0]
                            if language in self.meta_infos["task_language_mapping"][finetuning_task]:
                                if self.meta_infos["model_language_lookup_table"][_name_or_path] == language or \
                                        self.meta_infos["model_language_lookup_table"][_name_or_path] == 'all':
                                    predict_language_mean = self.get_average_score(finetuning_task=finetuning_task,
                                                                                   _name_or_path=_name_or_path,
                                                                                   score=aps)
                                    # This is to avoid string values; if there were no scores available I returned an
                                    # empty string, because 0.0 would be misleading
                                    if predict_language_mean not in ["", np.nan]:
                                        predict_language_mean_collected.append(predict_language_mean)
                                        predict_language_collected.append(language)
                                    else:
                                        logging.warning(
                                            f'There is an error for the fine-tuning task {finetuning_task} with the model {_name_or_path} and the score {aps}. '
                                            f'There are no useful values to calculate an mean score. ')

                        if len(predict_language_mean_collected) > 0:

                            predict_language_mean_collected = [plm[0] if isinstance(plm, tuple) else plm for plm in
                                                               predict_language_mean_collected]

                            mean_macro_f1_score = self.get_mean_from_list_of_values(predict_language_mean_collected,
                                                                                    with_standard_deviation=with_standard_deviation)

                            if set(predict_language_collected) != set(
                                    self.meta_infos["task_language_mapping"][finetuning_task]):
                                if set(self.meta_infos["model_language_lookup_table"][_name_or_path]) != set(
                                        predict_language_collected):
                                    logging.error(
                                        f"We do not have the prediction results for all languages for fine-tuning task {finetuning_task} with language model {_name_or_path}")
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

    def insert_aggregated_score_over_language_models(self, dataframe: pd.core.frame.DataFrame,
                                                     column_name: str = "Agg.") -> pd.core.frame.DataFrame:

        """
        For each language model it will collect all aggregated scores for each fine-tuning task/dataset and
        calculate the aggregate score for each language model.

        @param dataframe: pd.core.frame.DataFrame
        @param column_name: Name of the column that presents the aggregate score.
        @return: pd.core.frame.DataFrame
        """

        dataframe[column_name] = ""

        for _name_or_path in dataframe.index.tolist():
            columns = dataframe.columns.tolist()
            all_mean_macro_f1_scores = dataframe.loc[_name_or_path].tolist()
            all_mean_macro_f1_scores = [plm[0] if isinstance(plm, tuple) else plm for plm in all_mean_macro_f1_scores]
            all_mean_macro_f1_scores_cleaned = [x for x in all_mean_macro_f1_scores if type(x) != str]
            string_values_indices = [columns[all_mean_macro_f1_scores.index(x)] for x in all_mean_macro_f1_scores if
                                     type(x) == str]
            string_values_indices = list(set(string_values_indices))
            if all_mean_macro_f1_scores_cleaned != all_mean_macro_f1_scores:
                logging.warning(
                    f'Attention! {_name_or_path} has string values as mean score for the following datasets/languages: ' + ', '.join(
                        string_values_indices))

            all_mean_macro_f1_scores_mean = self.get_mean_from_list_of_values(all_mean_macro_f1_scores_cleaned)
            dataframe.at[_name_or_path, column_name] = all_mean_macro_f1_scores_mean

        return dataframe

    def get_dataset_aggregated_score(self, average_over_language=True, write_to_csv=True, task_constraint: list = [],
                                     model_constraint: list = []) -> None:

        """
        Creates the dataframe with the dataset aggregate scores.

        @param average_over_language: Boolean Value. If set to 'True', we will average over languages. Default is True.
        @param task_constraint: List of tasks that we want to check.
        @param model_constraint:  List of language models that we want to check.
        @return: None.
        """

        self.get_config_aggregated_score(
            average_over_language=average_over_language, write_to_csv=write_to_csv, model_constraint=model_constraint,
            task_constraint=task_constraint)

        columns = list()
        for dataset, configs in self.meta_infos["dataset_to_config"].items():
            if len(task_constraint) > 0:
                configs = [c for c in configs if c in task_constraint]
            if len(configs) > 0:
                columns.append(dataset)

        self.dataset_aggregated_score = self.create_template(columns=columns)

        relevant_models = self.results._name_or_path.unique()

        if len(model_constraint) > 0:
            relevant_models = [rm for rm in relevant_models if rm in model_constraint]

        for _name_or_path in relevant_models:
            for dataset in columns:
                configs = self.meta_infos["dataset_to_config"][dataset]
                list_for_dataset_mean = list()
                for conf in configs:
                    # Look if there are results for that config. If not, skip it.
                    if conf in self.config_aggregated_score.columns.tolist():
                        config_mean = self.config_aggregated_score.at[_name_or_path, conf]
                        if not isinstance(config_mean, float):
                            logging.warning(
                                f"There is no config mean for config {conf} with language model {_name_or_path}")
                        list_for_dataset_mean.append(config_mean)
                    else:
                        logging.warning(
                            f"There seem to be no results for the configuration {conf} with language model {_name_or_path}")
                if len(list_for_dataset_mean) > 0:
                    if len(list_for_dataset_mean) != len(configs):
                        logging.error(
                            f'Attention! It seems for dataset {dataset}' +
                            'you do not have the average values for all '
                            'configs/fine-tuning task. The average score will be '
                            'calculated on the basis of incomplete '
                            'information.')
                    dataset_mean = self.get_mean_from_list_of_values(list_for_dataset_mean)
                else:
                    dataset_mean = ''

                self.dataset_aggregated_score.at[_name_or_path, dataset] = dataset_mean

        self.dataset_aggregated_score = self.insert_aggregated_score_over_language_models(
            self.dataset_aggregated_score)

        if len(model_constraint) > 0:
            self.dataset_aggregated_score = self.dataset_aggregated_score[
                self.dataset_aggregated_score.index.isin(model_constraint)]

        if write_to_csv == True:
            dataset_aggregated_score = deepcopy(self.dataset_aggregated_score)
            dataset_aggregated_score = insert_abbreviations_in_columns(
                dataset_aggregated_score)
            dataset_aggregated_score = self.postprocess_columns(
                dataset_aggregated_score)
            self.write_to_files(dataframe_with_aggregate_scores=dataset_aggregated_score,
                                filename='dataset_aggregated_scores',
                                average_over_language=average_over_language)

    def get_language_aggregated_score(self, write_to_csv=True, task_constraint: list = [], model_constraint: list = []):
        all_languages = set()

        for languages in self.meta_infos['task_language_mapping'].values():
            for lang in languages:
                # TODO: Remove this constraint in the future
                if lang not in ['tr', 'nb']:
                    all_languages.add(lang)

        self.language_aggregated_score = self.create_template(all_languages)

        for score_type in self.available_predict_scores:
            language = score_type.split('_')[0]
            lookup_table = self.get_aggregated_score_for_language(score_type, task_constraint)
            for _name_or_path, score in lookup_table.items():
                self.language_aggregated_score.at[_name_or_path, language] = score

        self.language_aggregated_score = self.insert_aggregated_score_over_language_models(
            self.language_aggregated_score)

        if len(model_constraint) > 0:
            self.language_aggregated_score = self.language_aggregated_score[
                self.language_aggregated_score.index.isin(model_constraint)]

        # Order of columns
        column_order = sorted(self.language_aggregated_score.columns.tolist())
        column_order = [c for c in column_order if c != 'Agg.']
        self.language_aggregated_score = self.language_aggregated_score[column_order + ['Agg.']]

        language_aggregated_score = deepcopy(self.language_aggregated_score)
        language_aggregated_score = self.postprocess_columns(
            language_aggregated_score)

        if write_to_csv == True:
            # Extract "language_aggregated_scores" as variable or constant
            language_aggregated_score.to_csv(
                f'{self.output_dir}/language_aggregated_scores.csv')
            language_aggregated_score.style.highlight_max(color='lightgreen', axis=0).to_excel(
                f'{self.output_dir}/language_aggregated_scores.xlsx')
            make_latext_table(language_aggregated_score, f'{self.output_dir}/language_aggregated_scores.tex')

    def get_aggregated_score_for_language(self, score_type: str, task_constraint: list = [],
                                          with_standard_deviation: bool = False) -> dict:

        """
        For each score type the relevant model names are selected as well as the relevant finetuning tasks.
        Then, for each score type (language-specific score) the average of configs is calculated.
        Then, for each score type (language-specific score) the average over datasets (from the config aggregates) is calculated.

        @param score_type: Name of the score, such as de_predict/_macro-f1.
        @param task_constraint: List of tasks that we want to check.
        @param with_standard_deviation: Boolean value. Default is False. Whether you want to add standard deviation or not.
        @return: dict
        """

        tasks_relevant_for_language = list(self.results[(self.results[score_type].isnull() == False) & (
                self.results[score_type] != "")].finetuning_task.unique())

        if len(task_constraint) > 0:
            tasks_relevant_for_language = [t for t in tasks_relevant_for_language if t in task_constraint]

        languge_models_relevant_for_language = list(
            self.results[(self.results[score_type].isnull() == False) & (self.results[score_type] != "")][
                "_name_or_path"].unique())

        # Collect all average scores for each config grouped by dataset
        language_model_config_score_dict = defaultdict(dict)
        for finetuning_task in tasks_relevant_for_language:
            for _name_or_path in languge_models_relevant_for_language:
                result = self.get_average_score(finetuning_task=finetuning_task,
                                                _name_or_path=_name_or_path,
                                                score=score_type,
                                                with_standard_deviation=with_standard_deviation)
                if result not in ["", np.nan]:
                    dataset_for_finetuning_task = self.meta_infos['config_to_dataset'][finetuning_task]
                    if dataset_for_finetuning_task not in language_model_config_score_dict[_name_or_path].keys():
                        language_model_config_score_dict[_name_or_path][dataset_for_finetuning_task] = list()
                    language_model_config_score_dict[_name_or_path][dataset_for_finetuning_task].append(result)

        language_model_config_score_dict = dict(language_model_config_score_dict)

        # Average over configs per dataset
        results_averaged_over_configs = defaultdict(dict)

        for _name_or_path, dataset_and_config_scores in language_model_config_score_dict.items():
            for dataset, config_scores in dataset_and_config_scores.items():
                results_averaged_over_configs[_name_or_path][dataset] = self.get_mean_from_list_of_values(config_scores)

        results_averaged_over_configs = dict(results_averaged_over_configs)

        # Average over datasets within language model
        results_averaged_over_datasets = dict()

        for _name_or_path, dataset_and_score in results_averaged_over_configs.items():
            results_averaged_over_datasets[_name_or_path] = self.get_mean_from_list_of_values(
                list(dataset_and_score.values()))

        results_averaged_over_datasets = dict(results_averaged_over_datasets)

        return results_averaged_over_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose', help='Whether to enable verbose logging',
                        action='store_true', default=False)
    parser.add_argument('-pce', '--path_to_csv_export', help='Insert the path to the exported csv file from wandb',
                        default=None)
    # TODO consider renaming to just language
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
    parser.add_argument('-rs', '--report_spec',
                        help='This is the json file that contains the wandb project name and the tasks and models '
                             'that should be considered for the report.',
                        default=None)
    parser.add_argument('-los', '--list_of_seeds',
                        help='Decide what the required seeds are.',
                        default=[1, 2, 3])

    args = parser.parse_args()

    with open(TABLE_TO_REPORT_SPECS + f"/{args.report_spec}.json", 'r') as f:
        report_spec = js.load(f)

    if isinstance(args.list_of_seeds, str):
        list_of_seeds = args.list_of_seeds.split(',')
        list_of_seeds = [int(s.strip()) for s in list_of_seeds]
    else:
        list_of_seeds = args.list_of_seeds
    ra = ResultAggregator(wandb_api_key=args.wandb_api_key,
                          project_name=report_spec['wandb_project_name'],
                          output_dir=TABLE_OUTPUT_PATH + f'/{report_spec["wandb_project_name"]}',
                          path_to_csv_export=args.path_to_csv_export,
                          verbose_logging=args.verbose,
                          only_completed_tasks=False,
                          which_language=args.which_language,
                          required_seeds=list_of_seeds,
                          report_spec_name=args.report_spec,
                          fill_with_wrong_revisions=False,
                          mean_type='harmonic',
                          remove_outliers=True)

    ra.get_info()
    model_constraint = [_name_or_path.split(
        '@')[0] for _name_or_path in report_spec['_name_or_path']]
    ra.create_report(task_constraint=report_spec['finetuning_task'],
                     model_constraint=model_constraint,
                     only_completed_tasks=False)

    ra.get_dataset_aggregated_score(task_constraint=report_spec['finetuning_task'],
                                    model_constraint=model_constraint)

    ra.get_language_aggregated_score(task_constraint=report_spec['finetuning_task'],
                                     model_constraint=model_constraint)

    # Comment the method below out if you do not want config aggregate scores with standard deviation
    ra.get_config_aggregated_score(average_over_language=True,
                                   write_to_csv=True,
                                   model_constraint=model_constraint,
                                   task_constraint=report_spec['finetuning_task'],
                                   with_standard_deviation=True)

    with open(ra.output_dir + '/removed_outliers.json', 'w') as f:
        js.dump(ra.removed_values_dict, fp=f, indent=2)

    # TODO maybe move all of this reporting functionality into a separate folder

    # export KMP_DUPLICATE_LIB_OK=TRUE && python create_overview.py -wak 16faa77953e6003f2150513ada85c66660bdb0f9 -pn swiss-legal-data/neurips2023 -wl multilingual
