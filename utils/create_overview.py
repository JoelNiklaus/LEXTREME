from datetime import datetime
from pathlib import Path
from statistics import mean
import argparse
import json as js
import logging
import os
import pandas as pd
import re
import sys
import wandb_summarizer.download
import numpy as np
from collections import defaultdict



with open(os.path.abspath("../meta_infos.json"), "r") as f:
    meta_infos = js.load(f)


# TODO: Add a function for error analysis, i.e. which looks if there are enough seeds for each task + model

class ResultAggregator:


    meta_infos = meta_infos


    def __init__(self, wandb_api_key=None, path_to_csv_export=None):
        

        name_of_log_file = 'logs/loggings_'+datetime.now().isoformat()+'.txt'
        name_of_log_file = os.path.join(os.path.dirname(__file__), name_of_log_file)

        logging.basicConfig(level=logging.DEBUG, filename=name_of_log_file, filemode="a+",format="%(asctime)-15s %(levelname)-8s %(message)s")

        self.wandb_api_key = wandb_api_key
        
        if type(wandb_api_key)==str and bool(re.search('\d',wandb_api_key)):
            os.environ["WANDB_API_KEY"] = self.wandb_api_key

        self._path = name_of_log_file
        if path_to_csv_export is None:
            results = self.get_wandb_overview()
        else:
            results = pd.read_csv(Path(path_to_csv_export))
            results = self.edit_result_dataframe(results)
        results = results[results.finetuning_task.isnull()==False]

        # When fetching the data from the wandb api it it return state instead of State as column name
        # I could make every column name lowercase, but I am afraid this might cause some problems
        if 'State' in results.columns:
            self.results = results[results.State=="finished"]
        elif 'state' in results.columns:
            self.results = results[results.state=="finished"]

    def __enter__(self):
        sys.stdout = open(self._path, mode="w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    def get_info(self): 
        logging.info("The current export contains results for the following tasks:")
        logging.info(self.results.finetuning_task.unique())


    def get_wandb_overview(self, project_name:str=None):
        if project_name is None:
            project_name="lextreme/paper_results"
        results = wandb_summarizer.download.get_results(project_name=project_name)
        results = pd.DataFrame(results)
        results.to_csv("current_wandb_results_unprocessed.csv", index=False)
        results = self.edit_result_dataframe(results)
        return results

    def edit_column_name(self, column_name):
        if column_name.startswith('config_'):
            column_name = re.sub('^config_','', column_name)
        if column_name.startswith('end'):
            column_name = re.sub('^end_','', column_name)
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
            dataframe = dataframe.rename(columns={col:col_new})
        
        return dataframe

    def update_language_specifc_predict_columns(self, dataframe, score = None):
    
        """ For monolingual datasets we insert the overall macro-f1 score into the column for the language specific macro-f1 score """
        
        if score is None:
            score = 'predict/_macro-f1'
            
        all_available_languages_for_training_set = list(dataframe['language'].unique())
        all_available_languages_for_training_set = [l for l in all_available_languages_for_training_set if l!="all"]
        
        for language in all_available_languages_for_training_set:
            column_name = language + '_'+score
            indices = dataframe[dataframe.language==language].index.tolist()
            for i in indices:
                dataframe.at[i,column_name]=dataframe.at[i, score]
        
    
        return dataframe

    def edit_result_dataframe(self, results):
        results = self.edit_column_names_in_df(results)
        all_finetuning_tasks = results.finetuning_task 
        results['language']= all_finetuning_tasks.apply(lambda x: str(x).split('_')[0])
        results['finetuning_task']= all_finetuning_tasks.apply(lambda x: '_'.join(str(x).split('_')[1:]))
        results['hierarchical']=results.name.apply(self.insert_hierarchical)
        # Remove all cases where language == nan
        results = results[results.language!="nan"]
        results['seed'] = results.seed.apply(lambda x: int(x))
        results = self.update_language_specifc_predict_columns(results)

        return results
        

    def convert_numpy_float_to_python_float(self, value):
        if value == np.nan:
            return ""
        elif isinstance(value, np.floating):
            return value.item()
        else:
            return value

    def get_mean_from_list_of_values(self, list_of_values):
        list_of_values = [x for x in list_of_values if type(x)!=str]
        if len(list_of_values)>0:
            return self.convert_numpy_float_to_python_float(mean(list_of_values))
        else:
            return ""
    

    def get_average_score(self, finetuning_task:str, _name_or_path:str, score:str='predict/_macro-f1') -> float:
    
        results_filtered = self.results[(self.results.finetuning_task==finetuning_task) & (self.results._name_or_path==_name_or_path)][["seed",score]]

        if len(results_filtered.seed.unique())<3:
            logging.warning("Attention. For task "+ finetuning_task+" with model name "+_name_or_path+" you have "+str(len(results_filtered.seed.unique()))+" instead of 3 seeds! The mean will be calculated anway.")
        
        if len(results_filtered.seed.unique()) != len(results_filtered.seed.tolist()):
            logging.warning('Attention! It seems you have duplicate seeds for task '+finetuning_task+' with the model '+_name_or_path)
            logging.info('Duplicate values will be removed based on the seed number.')
            
            results_filtered.drop_duplicates('seed', inplace=True)
        
        if set(results_filtered.seed.unique()).intersection({1,2,3}) != {1,2,3}:
            logging.error('Attention! It seems you do not have the required seeds 1,2,3 for the finetuning task '+finetuning_task+' with the model '+_name_or_path+ '. The average score will be calculated on the basis of incomplete information.')

        if len(results_filtered.seed.tolist())==0:
            logging.info('It seems you have duplicate seeds task '+finetuning_task+' with the model '+_name_or_path+ " has no meaningful results.")
            return "" # There is nothing to be calculated
        else:
            mean_value = results_filtered[score].mean()
            if mean_value == np.nan:
                return ""
            else:
                return self.convert_numpy_float_to_python_float(mean_value)


    def create_template(self, columns = None):
        # Create empty dataframe
        overview_template = pd.DataFrame()

        # Se the list of model names as index
        overview_template = overview_template.set_axis(self.results._name_or_path.unique())

        # Name the index axis as Model
        overview_template.index.names = ['Model']

        if columns is None:
            for finetuning_task in self.results.finetuning_task.unique():
                overview_template[finetuning_task]=''
        else:
            for col in columns:
                overview_template[col]=''

        return overview_template


    def insert_aggregated_score_over_language_models(self, dataframe, column_name="aggregated_score"):
        
        dataframe[column_name]=""

        for _name_or_path in dataframe.index.tolist():
            all_mean_macro_f1_scores =  dataframe.loc[_name_or_path].tolist()
            all_mean_macro_f1_scores_cleaned = [x for x in all_mean_macro_f1_scores if type(x)!=str]
            if all_mean_macro_f1_scores_cleaned!=all_mean_macro_f1_scores:
                logging.warning('Attention! There are string values for your score!')

            all_mean_macro_f1_scores_mean = self.get_mean_from_list_of_values(all_mean_macro_f1_scores_cleaned)
            dataframe.at[_name_or_path,column_name]=all_mean_macro_f1_scores_mean

        columns = dataframe.columns.tolist()
        first_columns = [column_name]
        dataframe = dataframe[first_columns+[col for col in columns if col not in first_columns]]

        return dataframe


    def insert_config_average_scores(self, overview_template, available_predict_scores=None):

        logging.info("*** Calculating average scores ***")

        if available_predict_scores is None:
            for _name_or_path in self.results._name_or_path.unique():
                for finetuning_task in self.results.finetuning_task.unique():
                    mean_macro_f1_score = self.get_average_score(finetuning_task, _name_or_path)
                    overview_template.at[_name_or_path,finetuning_task]=mean_macro_f1_score

        else:
            for _name_or_path in self.results._name_or_path.unique():
                for finetuning_task in self.results.finetuning_task.unique():
                    
                    # We check of the finetuning task has more than one language
                    # If not, we can process with the normal predict/_macro-f1 that stands for the entire config
                    # If yes, we loop through all avalaible language-specific macro-f1 scores

                    if len(self.meta_infos["task_language_mapping"][finetuning_task])==1:
                        mean_macro_f1_score = self.get_average_score(finetuning_task, _name_or_path)
                    
                    elif len(self.meta_infos["task_language_mapping"][finetuning_task])>1:
                        predict_language_mean_collected = list()
                        predict_language_collected = list()

                        # Average over languages
                        for aps in available_predict_scores:
                            language = aps.split('_')[0]
                            predict_language_mean = self.get_average_score(finetuning_task,_name_or_path,aps)
                            if predict_language_mean not in ["", np.nan]: # This is to avoid string values; if there were no scores available I returned an empty string, because 0.0 would be missleading
                                predict_language_mean_collected.append(predict_language_mean)
                                predict_language_collected.append(language)
                            #else:
                                # TODO: Add logger informartion
                                
                        if len(predict_language_mean_collected)>0:
                            if set(predict_language_collected) != set(self.meta_infos["task_language_mapping"][finetuning_task]):
                                logging.error("We do not have the prediction results for all languages for finetuning task "+finetuning_task+" with language model " +_name_or_path)
                                logging.info("We have results for the following languages: " + ", ".join(sorted(predict_language_collected)))
                                logging.info("But we need results for the following languages: " + ", ".join(sorted(self.meta_infos["task_language_mapping"][finetuning_task])))
                                logging.info("The the results for the following language(s) are mssing: " + ', '.join([l for l in self.meta_infos["task_language_mapping"][finetuning_task] if l not in predict_language_collected]))

                            mean_macro_f1_score = self.get_mean_from_list_of_values(predict_language_mean_collected)
                        else:
                            mean_macro_f1_score = ""
                            
                    overview_template.at[_name_or_path,finetuning_task]=mean_macro_f1_score
                        


        #if overview_template.isnull().values.any():
            #logging.warning('Attention! For some cases we do not have an aggregated score! These cases will be converted to nan.')
            #overview_template.fillna("", inplace=True)

        
        overview_template = overview_template



    def get_config_aggregated_score(self, average_over_language=True, write_to_csv=False):

        # Insert aggregated mean for each model name

        self.config_aggregated_score = self.create_template()

        available_predict_scores = [col for col in self.results if bool(re.search(r'^\w+_predict/_macro-f1',col))]
        
        if average_over_language==False:
            self.insert_config_average_scores(self.config_aggregated_score)
        elif average_over_language == True:
            self.insert_config_average_scores(self.config_aggregated_score, available_predict_scores)

        self.config_aggregated_score = self.insert_aggregated_score_over_language_models(self.config_aggregated_score)
        
        if write_to_csv:
            if average_over_language==False:
                self.config_aggregated_score.to_csv('config_aggregated_scores_simple.csv')
            if average_over_language==True:
                self.config_aggregated_score.to_csv('config_aggregated_scores_average_over_language.csv')


    def get_dataset_aggregated_score(self, average_over_language=True, write_to_csv=True, task_constraint=None):

        self.get_config_aggregated_score(average_over_language=average_over_language, write_to_csv=write_to_csv)

        columns = list(self.meta_infos["dataset_to_config"].keys())

        self.dataset_aggregated_score = self.create_template(columns=columns)

        for _name_or_path in self.results._name_or_path.unique():
            for dataset, configs in self.meta_infos["dataset_to_config"].items():
                if task_constraint is not None:
                    configs = [conf for conf in configs if conf in task_constraint]
                    
                dataset_mean = list()
                for conf in configs:
                    config_mean = self.config_aggregated_score.at[_name_or_path, conf]
                    if config_mean=="": # This is to avoid string values; if there were no scores available I returned an empty string, because 0.0 would be missleading
                        logging.info("There is no config mean for config "+conf+" with language model "+_name_or_path)
                    elif type(config_mean)==float:
                        dataset_mean.append(config_mean)
                    else:
                        logging.error("Processed interrupted due to wrong mean value that is not a float. The mean value is: " + str(config_mean))
                        break
                if len(dataset_mean)>0:
                    if len(dataset_mean)!=len(configs):
                        logging.error('Attention! It seems for dataset ' +dataset + ' you do not have the average values for configs. The average score will be calculated on the basis of incomplete information.')
                    dataset_mean = self.get_mean_from_list_of_values(dataset_mean)
                else:
                    dataset_mean = ''
                self.dataset_aggregated_score.at[_name_or_path, dataset] = dataset_mean

        self.dataset_aggregated_score = self.insert_aggregated_score_over_language_models(self.dataset_aggregated_score)
    
        if write_to_csv:
            if average_over_language==False:
                self.dataset_aggregated_score.to_csv('dataset_aggregated_scores_simple.csv')
            if average_over_language==True:
                self.dataset_aggregated_score.to_csv('dataset_aggregated_scores_average_over_language.csv')


    
    def get_aggregated_score_for_language(self, score_type, task_constraint=None):
        
        tasks_relevant_for_language = list(self.results[self.results[score_type].isnull()==False].finetuning_task.unique())

        if task_constraint is not None:
            tasks_relevant_for_language = [t for t in tasks_relevant_for_language if t in task_constraint]

        languge_models_relevant_for_language = list(self.results[self.results[score_type].isnull()==False]["_name_or_path"].unique())
        tasks_relevant_for_language


        # Collect all average scores for each config grouped by dataset
        language_model_config_score_dict = defaultdict(dict)
        for ft in tasks_relevant_for_language: 
            for lm in languge_models_relevant_for_language:
                result = self.get_average_score(ft, lm, score_type)
                if result not in ["", np.nan]:
                    dataset_for_finetuning_task = self.meta_infos['config_to_dataset'][ft]
                    if dataset_for_finetuning_task not in language_model_config_score_dict[lm].keys():
                        language_model_config_score_dict[lm][dataset_for_finetuning_task]=list()
                    language_model_config_score_dict[lm][dataset_for_finetuning_task].append(result)


        language_model_config_score_dict = dict(language_model_config_score_dict)

        # Average over configs per dataset
        results_averaged_over_configs = defaultdict(dict)

        for language_model, dataset_and_config_scores in language_model_config_score_dict.items():
            for dataset, config_scores in dataset_and_config_scores.items():
                results_averaged_over_configs[language_model][dataset]=self.get_mean_from_list_of_values(config_scores)

        results_averaged_over_configs = dict(results_averaged_over_configs)


        # Average over datasets within language model

        results_averaged_over_datasets = dict()

        for language_model, dataset_and_score in results_averaged_over_configs.items():
            results_averaged_over_datasets[language_model]=self.get_mean_from_list_of_values(list(dataset_and_score.values()))

        results_averaged_over_datasets = dict(results_averaged_over_datasets)
        
        return results_averaged_over_datasets

    
    def get_language_aggregated_score(self, write_to_csv=True, task_constraint=None):

        all_languages = set()

        for languages in self.meta_infos['task_language_mapping'].values():
            for l in languages:
                all_languages.add(l)

        self.language_aggregated_score = self.create_template(all_languages)
        
        available_predict_scores = [col for col in self.results if bool(re.search(r'^\w+_predict/_macro-f1',col))]
        available_predict_scores = sorted(available_predict_scores)

        for score_type in available_predict_scores:
            language = score_type.split('_')[0]
            lookup_table = self.get_aggregated_score_for_language(score_type, task_constraint)
            for language_model, score in lookup_table.items():
                self.language_aggregated_score.at[language_model,language]=score

        self.language_aggregated_score = self.insert_aggregated_score_over_language_models(self.language_aggregated_score)

        if write_to_csv:
            self.language_aggregated_score.to_csv('language_aggregated_scores.csv')


        
        

    
    
            
            









if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-pce', '--path_to_csv_export', help='Insert the path to the exported csv file from wandb', default=None)
    parser.add_argument('-wak', '--wandb_api_key', help='To be able to fetch the right results, you can insert the wand api key. Alternatively, set the WANDB_API_KEY environment variable to your API key.', default=None)

    args = parser.parse_args()

    ra = ResultAggregator(wandb_api_key=args.wandb_api_key)

    ra.get_info()

    ra.get_dataset_aggregated_score()

    



