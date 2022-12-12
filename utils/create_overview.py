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



with open(os.path.abspath("../meta_infos.json"), "r") as f:
    meta_infos = js.load(f)



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
        results['language']=results.finetuning_task.apply(lambda x: str(x).split('_')[0])
        results['finetuning_task']=results.finetuning_task.apply(lambda x: '_'.join(str(x).split('_')[1:]))
        results['hierarchical']=results.name.apply(self.insert_hierarchical)
        # Remove all cases where language == nan
        results = results[results.language!="nan"]
        results['seed'] = results.seed.apply(lambda x: int(x))
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

    def get_average_score(self, finetuning_task:str, _name_or_path:str, score:str='predict/_macro-f1') -> float:
    
        results_filtered = self.results[(self.results.finetuning_task==finetuning_task) & (self.results._name_or_path==_name_or_path)][["seed",score]]

        if len(results_filtered.seed.unique())<3:
            logging.warning("Attention. For task "+ finetuning_task+"with model name "+_name_or_path+" you have "+str(len(results_filtered.seed.unique()))+" instead of 3 seeds! The mean will be calculated anway.")
        
        if len(results_filtered.seed.unique()) != len(results_filtered.seed.tolist()):
            logging.warning('Attention! It seems you have duplicate seeds for task '+finetuning_task+' with the model '+_name_or_path)
            logging.info('Duplicate values will be removed based on the seed number.')
            
            results_filtered.drop_duplicates('seed', inplace=True)
        
        if set(results_filtered.seed.unique()).intersection({1,2,3}) != {1,2,3}:
            logging.error('Attention! It seems you do not have the required seeds 1,2,3 for the finetuning task '+finetuning_task+' with the model '+_name_or_path+ '. The average score will be calculated on the basis of incomplete information.')
        
        return results_filtered[score].mean()


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
                    if len(meta_infos["task_language_mapping"][finetuning_task])==1:
                        mean_macro_f1_score = self.get_average_score(finetuning_task, _name_or_path)
                        overview_template.at[_name_or_path,finetuning_task]=mean_macro_f1_score
                    
                    elif len(meta_infos["task_language_mapping"][finetuning_task])>1:
                        predict_language_mean_collected = list()
                        predict_language_collected = list()

                        # Average over languages
                        for aps in available_predict_scores:
                            language = aps.split('_')[0]
                            predict_language_mean = self.get_average_score(finetuning_task,_name_or_path,aps)
                            if str(predict_language_mean)!="nan": # This is to avoid nans
                                predict_language_mean_collected.append(predict_language_mean)
                                predict_language_collected.append(language)
                        if len(predict_language_mean_collected)>0:
                            if set(predict_language_collected)!=set(meta_infos["task_language_mapping"][finetuning_task]):
                                logging.error("We do not have the prediction results for all languages for finetuning task "+finetuning_task+" with language model " +_name_or_path)
                                logging.log("We have results for the following languages: " + ", ".join(sorted(predict_language_collected)))
                                logging.log("But we need results for the following languages: " + ", ".join(sorted(meta_infos["task_language_mapping"][finetuning_task])))
                                logging.log("The the results for the following language(s) are mssing: " + ', '.join([l for l in meta_infos["task_language_mapping"][finetuning_task] if l not in predict_language_collected]))

                            mean_macro_f1_score = mean(predict_language_mean_collected)
                        else:
                            mean_macro_f1_score = 0.0
                            
                        overview_template.at[_name_or_path,finetuning_task]=mean_macro_f1_score
                        


        if overview_template.isnull().values.any():
            logging.warning('Attention! For some cases we do not have an aggregated score! These cases will be converted to nan.')
            overview_template.fillna(0.0, inplace=True)



    def get_config_aggregated_score(self, average_over_language=True, write_to_csv=False):

        # Insert aggregated mean for each model name

        self.config_aggregated_score = self.create_template()

        available_predict_scores = [col for col in self.results if bool(re.search(r'^\w+_predict/_macro-f1',col))]
        
        if average_over_language==False:
            self.insert_config_average_scores(self.config_aggregated_score)
        elif average_over_language == True:
            self.insert_config_average_scores(self.config_aggregated_score, available_predict_scores)

        self.config_aggregated_score['config_aggregate_score']=''

        for _name_or_path in self.config_aggregated_score.index.tolist():
            all_mean_macro_f1_scores =  self.config_aggregated_score.loc[_name_or_path].tolist()
            all_mean_macro_f1_scores_cleaned = [x for x in all_mean_macro_f1_scores if type(x)!=str]
            if all_mean_macro_f1_scores_cleaned!=all_mean_macro_f1_scores:
                logging.warning('Attention! There are string values for your score!')
                
            all_mean_macro_f1_scores_mean = mean(all_mean_macro_f1_scores_cleaned)
            self.config_aggregated_score.at[_name_or_path,'config_aggregate_score']=all_mean_macro_f1_scores_mean

        columns = self.config_aggregated_score.columns.tolist()
        first_columns = ['config_aggregate_score']
        self.config_aggregated_score = self.config_aggregated_score[first_columns+[col for col in columns if col not in first_columns]]
        
        if write_to_csv:
            if average_over_language==False:
                self.config_aggregated_score.to_csv('config_aggregated_scores_simple.csv')
            if average_over_language==True:
                self.config_aggregated_score.to_csv('config_aggregated_scores_average_over_language.csv')


    def get_dataset_aggregated_score(self, average_over_language=True, write_to_csv=True):

        self.get_config_aggregated_score(average_over_language=average_over_language, write_to_csv=write_to_csv)

        columns = list(meta_infos["dataset_to_config"].keys())

        self.dataset_aggregated_score = self.create_template(columns=columns)

        for _name_or_path in self.results._name_or_path.unique():
            for dataset, configs in meta_infos["dataset_to_config"].items():
                dataset_mean = list()
                for conf in configs:
                    config_mean = self.config_aggregated_score.at[_name_or_path, conf]
                    if type(config_mean)!=str:
                        config_mean = config_mean.item() # convert numpy float to python float
                    if type(config_mean)==float:
                        dataset_mean.append(config_mean)
                if len(dataset_mean)>0:
                    if len(dataset_mean)!=len(configs):
                        logging.error('Attention! It seems for dataset ' +dataset + ' you do not have the average values for configs. The average score will be calculated on the basis of incomplete information.')
                    dataset_mean = mean(dataset_mean)
                else:
                    dataset_mean = ''
                self.dataset_aggregated_score.at[_name_or_path, dataset] = dataset_mean

        
        self.dataset_aggregated_score['dataset_aggregate_score']=''

        for _name_or_path in self.dataset_aggregated_score.index.tolist():
            dataset_all_scores = self.dataset_aggregated_score.loc[_name_or_path].tolist()
            dataset_all_scores = [float(x) for x in dataset_all_scores if x != 0.0 and x != ""]
            if len(dataset_all_scores)>0:
                dataset_aggregate_score = mean(dataset_all_scores)
            else:
                dataset_aggregate_score = 0.0
            
            self.dataset_aggregated_score.at[_name_or_path, 'dataset_aggregate_score'] = dataset_aggregate_score

        columns = self.dataset_aggregated_score.columns.tolist()
        first_columns = ['dataset_aggregate_score']
        self.dataset_aggregated_score = self.dataset_aggregated_score[first_columns+[col for col in columns if col not in first_columns]]



        if write_to_csv:
            if average_over_language==False:
                self.dataset_aggregated_score.to_csv('dataset_aggregated_scores_simple.csv')
            if average_over_language==True:
                self.dataset_aggregated_score.to_csv('dataset_aggregated_scores_average_over_language.csv')







        









if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-pce', '--path_to_csv_export', help='Insert the path to the exported csv file from wandb', default=None)
    parser.add_argument('-wak', '--wandb_api_key', help='To be able to fetch the right results, you can insert the wand api key. Alternatively, set the WANDB_API_KEY environment variable to your API key.', default=None)

    args = parser.parse_args()

    ra = ResultAggregator(wandb_api_key=args.wandb_api_key)

    ra.get_info()

    ra.get_dataset_aggregated_score()

    



