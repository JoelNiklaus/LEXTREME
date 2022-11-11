# Useful links

# Handling outliers in histograms: https://stackoverflow.com/questions/15837810/making-pyplot-hist-first-and-last-bins-include-outliers
# Multiple histograms in one plot: https://stackoverflow.com/questions/6871201/plot-two-histograms-on-single-chart-with-matplotlib

import pandas as pd
from datasets import load_dataset, Dataset
from matplotlib import pyplot as plt
import numpy as np
from ast import literal_eval
from transformers import AutoTokenizer, XLMRobertaTokenizer
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


#from pandarallel import pandarallel

#pandarallel.initialize(progress_bar=False,nb_workers=15)



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
                     'mapa_fine',
                     'multi_eurlex_level_1']





def get_tokenizers(list_of_models):
    tokenizer_dict = dict()
    
    for language_model in list_of_models:
        if language_model=="microsoft/Multilingual-MiniLM-L12-H384":
            tokenizer_dict[language_model]=XLMRobertaTokenizer.from_pretrained(language_model)
        else:
            tokenizer_dict[language_model]=AutoTokenizer.from_pretrained(language_model)

    
    return tokenizer_dict

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

    dataframe_as_dataset = Dataset.from_pandas(dataframe)
    print(dataframe_as_dataset)

    dataframe_processed = list()
    
    for language_model_name in models_to_be_used:
        dataset = deepcopy(dataframe_as_dataset)
        tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        dataset = dataset.map(lambda examples: tokenizer(examples["input"],add_special_tokens=False, return_length=True), batched=True, batch_size=50000, num_proc=25)
        dataset = dataset.rename_column("length", language_model_name)
        #dataset = dataset.remove_columns(['label', 'input_ids', 'attention_mask'])
        dataframe_processed.append(pd.DataFrame(dataset))

    dataframe_processed = pd.concat(dataframe_processed, axis = 1)
    dataframe_processed = dataframe_processed.loc[:,~dataframe_processed.columns.duplicated()] #Remove duplicate columns
    print(dataframe_processed.head())
    return dataframe_processed


def merge_equal_tokenization_outputs(dataframe,models_to_be_used):


    '''Some pretrained models will produce the same tokenization output. This function will group these outs together so that we have fewer histograms.'''
    
    interim_dict = dict()
    
    for language_model_name in models_to_be_used:
        tokenization_output = dataframe[language_model_name].values.tolist()
        interim_dict[language_model_name]=tokenization_output
        
    language_model_dict = deepcopy(interim_dict)
    
    language_models_that_generate_equal_outputs = set()
    
    for language_model_name in models_to_be_used:
        tokenization_output = dataframe[language_model_name].values.tolist()
        for key in interim_dict.keys():
            if key!=language_model_name:
                if interim_dict[key]==tokenization_output:
                    language_models_that_generate_equal_outputs.add(key)
                    language_models_that_generate_equal_outputs.add(language_model_name)
                    
    language_models_that_generate_equal_outputs = list(language_models_that_generate_equal_outputs)
    print(language_models_that_generate_equal_outputs)
    
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


def split_into_languages(dataset_df):

    '''This function is only used for the multi_eurlex datasets to split the dictionaries into languages'''

    dataset_new = list()

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
    if 'multi_eurlex' in dataset_name:
        all_data_as_df = split_into_languages(all_data_as_df)


    all_data_as_df = add_column_with_text_length(all_data_as_df,models_to_be_used)
    all_data_as_df = merge_equal_tokenization_outputs(all_data_as_df,models_to_be_used)
    
    return all_data_as_df

def processing_function_multi_eurlex(item, tokenizer_dict):

    dataset_new = list()


    for language, document in literal_eval(str(item['input'])).items():
        if document is not None:
            item_new = dict()
            item_new['language']=language
            item_new['input']=str(document)
            for language_model_name, tokenizer in tokenizer_dict.items():
                length = tokenizer(str(document),add_special_tokens=False, return_length=True)['length']
                if type(length)==int:
                    item_new[language_model_name]=length
                elif type(length)==list:
                    item_new[language_model_name]=length[0]
    
            dataset_new.append(item_new)

    return  dataset_new


def generate_dataframe_with_tokenization_for_multi_eurlex(dataset_name):

    '''This is used only for multi_eurlex_level_1, multi_eurlex_level_2, multi_eurlex_level_3'''

    if dataset_name.startswith("multi_eurlex_level"):

        tokenizer_dict = get_tokenizers(models_to_be_used)

        results_collected = list()


        for split in ["train","validation","test"]:
            
            dataset = load_dataset("joelito/lextreme", dataset_name, split=split) #streaming=True 
            dataset = dataset.select([n for n in range(0,100)])
            
            for example in dataset:
                example = processing_function_multi_eurlex(example, tokenizer_dict)
                for e in example:
                    results_collected.append(e)
        
        results_collected_df = pd.DataFrame(results_collected)
        
        return pd.DataFrame(results_collected_df)


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
        p[-1].set_alpha(1)


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




def generate_histograms(dataset_name, less_resources=True):

    '''Generates histogram per dataset and language'''
    
    if dataset_name.startswith("multi_eurlex_level") and less_resources==True:
        df_all = generate_dataframe_with_tokenization_for_multi_eurlex(dataset_name)
    else:
        df_all = generate_dataframe_with_tokenization(dataset_name)

    print(df_all.head())
    
    all_languages = df_all.language.unique()

    if len(all_languages)>1:
        for lang in all_languages:
            df = df_all[df_all.language==lang]
            draw_histogram(df, dataset_name, lang)
        draw_histogram(df_all, dataset_name)
    else:
        draw_histogram(df_all, dataset_name)


if __name__=='__main__':
    for dataset_name in lextreme_datasets:
        print('Processing: ',dataset_name)
        generate_histograms(dataset_name=dataset_name)
        print('\n#####################################################\n')
