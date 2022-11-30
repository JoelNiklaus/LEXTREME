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





class ResultAggregator:


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
        results = self.edit_column_names_in_df(results)
        return results

    def edit_column_name(self, column_name):
        if column_name.startswith('config_'):
            column_name = re.sub('^config_','', column_name)
        if column_name.startswith('end'):
            column_name = re.sub('^end_','', column_name)
        return column_name

    def edit_column_names_in_df(self, dataframe):
        columns = dataframe.columns.tolist()
        
        for col in columns:
            col_new = self.edit_column_name(col)
            dataframe = dataframe.rename(columns={col:col_new})
        
        return dataframe

    def get_average_score(self, finetuning_task:str, _name_or_path:str, score:str='predict/_macro-f1') -> float:
    
        results_filtered = self.results[(self.results.finetuning_task==finetuning_task) & (self.results._name_or_path==_name_or_path)][["seed",score]]

        if len(results_filtered.seed.unique())<5:
            logging.warning("Attention. For task "+ finetuning_task+"with model name "+_name_or_path+" you have "+str(len(results_filtered.seed.unique()))+" instead of 5 seeds! The mean will be calculated anway.")
        
        if len(results_filtered.seed.unique()) != len(results_filtered.seed.tolist()):
            logging.warning('Attention! It seems you have duplicate seeds for task '+finetuning_task+' with the model '+_name_or_path)
            logging.info('Duplicate values will be removed based on the seed number.')
            
            results_filtered.drop_duplicates('seed', inplace=True)
        
        return results_filtered[score].mean()


    def create_template(self):
        # Create empty dataframe
        overview_template = pd.DataFrame()

        # Se the list of model names as index
        overview_template = overview_template.set_axis(self.results._name_or_path.unique())

        # Name the index axis as Model
        overview_template.index.names = ['Model']

        for finetuning_task in self.results.finetuning_task.unique():
            overview_template[finetuning_task]=''

        self.overview_template = overview_template


    def insert_dataset_average_scores(self):

        logging.info("*** Calculating average scores ***")
        for _name_or_path in self.results._name_or_path.unique():
            for finetuning_task in self.results.finetuning_task.unique():
                mean_macro_f1_score = self.get_average_score(finetuning_task, _name_or_path)
                self.overview_template.at[_name_or_path,finetuning_task]=mean_macro_f1_score


        if self.overview_template.isnull().values.any():
            logging.warning('Attention! For some cases we do not have an aggregated score! These cases will be converted to zero.')
            self.overview_template.fillna(0.0, inplace=True)


    def get_dataset_aggregated_score(self):

        # Insert aggregated mean for each model name

        self.create_template()
        self.insert_dataset_average_scores()

        self.overview_template['dataset_aggregate_score']=''

        for _name_or_path in self.overview_template.index.tolist():
            all_mean_macro_f1_scores =  self.overview_template.loc[_name_or_path].tolist()
            all_mean_macro_f1_scores_cleaned = [x for x in all_mean_macro_f1_scores if type(x)!=str]
            if all_mean_macro_f1_scores_cleaned!=all_mean_macro_f1_scores:
                logging.warning('Attention! There are string values for your score!')
                
            all_mean_macro_f1_scores_mean = mean(all_mean_macro_f1_scores_cleaned)
            self.overview_template.at[_name_or_path,'dataset_aggregate_score']=all_mean_macro_f1_scores_mean

        columns = self.overview_template.columns.tolist()
        first_columns = ['dataset_aggregate_score']
        self.overview_template = self.overview_template[first_columns+[col for col in columns if col not in first_columns]]
        self.overview_template.to_csv('dataset_aggregated_scores.csv')







if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-pce', '--path_to_csv_export', help='Insert the path to the exported csv file from wandb', default=None)
    parser.add_argument('-wak', '--wandb_api_key', help='To be able to fetch the right results, you can insert the wand api key. Alternatively, set the WANDB_API_KEY environment variable to your API key.', default=None)

    args = parser.parse_args()

    ra = ResultAggregator(wandb_api_key=args.wandb_api_key)

    ra.get_info()

    ra.get_dataset_aggregated_score()

    



