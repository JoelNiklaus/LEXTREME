# Useful links

# Handling outliers in histograms: https://stackoverflow.com/questions/15837810/making-pyplot-hist-first-and-last-bins-include-outliers
# Multiple histograms in one plot: https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib

import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
from ast import literal_eval
from transformers import AutoTokenizer
import os
import shutil
from pathlib import Path
import re
from multiprocessing import Pool
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
from typing import List


plt.style.use('ggplot')


from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)



models_to_be_used = [
    "distilbert-base-multilingual-cased",
    "microsoft/Multilingual-MiniLM-L12-H384",
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base",
    "xlm-roberta-large"
    ]

lextreme_datasets = ['brazilian_court_decisions_judgment', 
                    'brazilian_court_decisions_unanimity', 
                     'swiss_judgment_prediction', 
                     'german_argument_mining', 
                     'greek_legal_code_volume', 
                     'greek_legal_code_chapter', 
                     'greek_legal_code_subject', 
                     'online_terms_of_service_unfairness_levels', 
                     'online_terms_of_service_clause_topics', 
                     'covid19_emergency_event', 
                     'lener_br', 
                     'legalnero', 
                     'greek_legal_ner', 
                     'mapa_coarse', 
                     'mapa_fine'] #'multi_eurlex_level_1'


lextreme_datasets = [
                    'brazilian_court_decisions_judgment', 
                     'mapa_fine'
                     ]




def get_tokenization_length(tokenizer, text):

    '''Returns a integer describing the length of the input text'''

    if type(text)==str:
        result = tokenizer(text,add_special_tokens=False, return_length=True)
    if type(text) in (list, tuple):
        if len(text)==0:
            result = dict()
            result['length']=0
        else:
            text = [str(x) for x in text]
            try:
                result = tokenizer(text, is_split_into_words=True,add_special_tokens=False, return_length=True)
            except:
                print(text)
    if type(result['length'])==int:
        return result['length']
    elif type(result['length'])==list:
        return result['length'][0]



def add_column_with_text_length(dataframe,models_to_be_used):

    '''Applies the function get_tokenization_length to the input text in a dataframe'''
    
    for language_model_name in models_to_be_used:
        tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        dataframe[language_model_name]=dataframe.input.apply(lambda x: get_tokenization_length(tokenizer,x))
        dataframe[language_model_name]=dataframe.input.apply(lambda x: get_tokenization_length(tokenizer,x))
    
    return dataframe


def merge_equal_tokenization_outputs(dataframe,models_to_be_used):


    '''Some pretrained models will produce the same tokenization output. This function will group these outs together so that we have fewer histograms.'''
    
    interim_dict = dict()
    
    for language_model_name in models_to_be_used:
        tokenization_output = dataframe[language_model_name].tolist()
        interim_dict[language_model_name]=tokenization_output
        
    language_model_dict = deepcopy(interim_dict)
    
    language_models_that_generate_equal_outputs = set()
    
    for language_model_name in models_to_be_used:
        tokenization_output = dataframe[language_model_name].tolist()
        for key in interim_dict.keys():
            if key!=language_model_name:
                if interim_dict[key]==tokenization_output:
                    language_models_that_generate_equal_outputs.add(key)
                    language_models_that_generate_equal_outputs.add(language_model_name)
                    
    language_models_that_generate_equal_outputs = list(language_models_that_generate_equal_outputs)
    
    dataframe[' and '.join(sorted(list(language_models_that_generate_equal_outputs)))]=dataframe[language_models_that_generate_equal_outputs[0]]
    
    for language_model_name in models_to_be_used:
        if language_model_name in language_models_that_generate_equal_outputs:
            del dataframe[language_model_name]
    
        
    return dataframe


def get_percentiles(list_with_values:List[int], percentile=99):

    '''Gets the values that fall outs the x-th percentile which will be grouped in the last bin as outliers in the histograms'''
    
    border = round(np.percentile(sorted(list_with_values),percentile))
    
    outliers = [x for x in list_with_values if x > border]
    
    outliers_sum = sum(outliers)
    
    list_with_values_new = [x for x in list_with_values if x <= border]
    
    return list_with_values_new, outliers


def split_into_languages(dataset):

    '''This function is only used for the multi_eurlex datasets to split the dictionaries into languages'''

    dataset_new = list()
    
    dataset_df = pd.DataFrame(dataset)
    
    for item in dataset_df.to_dict(orient='records'):
        labels = item['label']
        for language, document in literal_eval(item['input']).items():
            if document is not None:
                item_new = dict()
                item_new['language']=language
                item_new['input']=str(document)
                item_new['label']=labels
                dataset_new.append(item_new)
    
    dataset_new = pd.DataFrame(dataset_new)
    
    return  dataset_new


def generate_dataframe_with_tokenization(dataset_name):

    '''This function will fetch the dataset, convert it into a dataframe and apply tokenization'''

    dataset = load_dataset("joelito/lextreme",dataset_name) 

    all_data_as_df = list()

    for split in ['train','validation','test']:
        df = pd.DataFrame(dataset[split])
        all_data_as_df.append(df)

    all_data_as_df = pd.concat(all_data_as_df)
    all_data_as_df = all_data_as_df


    all_data_as_df = add_column_with_text_length(all_data_as_df,models_to_be_used)
    all_data_as_df = merge_equal_tokenization_outputs(all_data_as_df,models_to_be_used)
    
    return all_data_as_df


def draw_histogram(dataframe,dataset_name,language=None):

    plt.figure(figsize=(20,10))

    color_mapper = ['red','green','blue','grey','yellow']

    max_value = 0
    columns_with_values = defaultdict(dict)
    for c in dataframe.columns:
        for lmt in models_to_be_used:
            if re.search(lmt,c):
                values = dataframe[c].tolist()
                values, outliers = get_percentiles(values)
                columns_with_values[c]['values']=values
                columns_with_values[c]['outliers']=outliers

                if max(values)>max_value:
                    max_value=max(values) # We need the maximal value for the bin


    bins = np.linspace(0, max_value, 50)
    
    columns_with_values_keys = list(columns_with_values.keys())
    
    all_values = list()
    all_labels = list()

    for n, key in enumerate(columns_with_values_keys):
        
        values = columns_with_values[key]['values']
        
        all_values.append(values)
        all_labels.append(key)

            
    N, bins, patches = plt.hist(all_values, bins=bins, alpha=0.7, edgecolor='black', label=all_labels)
    
    for n,p in enumerate(patches):
        # The last bin will contain all outliers
        p[-1].set_height(p[-1].get_height()+len(columns_with_values[columns_with_values_keys[n]]['outliers']))


    plt.legend(loc='upper right')

    if language is not None:
        plt.title('Histogram for dataset '+dataset_name+' for input with language: '+language)
    else:
        plt.title('Histogram for dataset '+dataset_name)
    
    output_directory_name = 'histograms/'+dataset_name
    if os.path.isdir(output_directory_name)==False:
        os.makedirs(output_directory_name)
    
    if language is not None:
        plt.savefig(output_directory_name+'/'+dataset_name+'_language__'+language+'.jpg')
    else:
        plt.savefig(output_directory_name+'/'+dataset_name+'.jpg')




def generate_histograms(dataset_name):

    '''Generates histogram per dataset and language'''
    
    df_all = generate_dataframe_with_tokenization(dataset_name)
    
    all_languages = df_all.language.unique()

    if len(all_languages)>0:
        for lang in all_languages:
            df = df_all[df_all.language==lang]
            draw_histogram(df, dataset_name, lang)

    draw_histogram(df_all, dataset_name)


if __name__=='__main__':
    for dataset_name in lextreme_datasets[:3]:
        generate_histograms(dataset_name=dataset_name)
