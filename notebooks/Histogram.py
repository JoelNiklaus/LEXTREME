#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

plt.style.use('ggplot')
#plt.style.use('seaborn-deep')


# In[2]:


from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False)


# In[3]:


def get_tokenization_length(tokenizer, text):
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


# In[4]:


models_to_be_used = [
    "distilbert-base-multilingual-cased",
    "microsoft/Multilingual-MiniLM-L12-H384",
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base",
    "xlm-roberta-large"
    ]


# In[5]:


lextreme_datasets = ['brazilian_court_decisions_judgment', 'brazilian_court_decisions_unanimity', 
                     'swiss_judgment_prediction', 'german_argument_mining', 'greek_legal_code_volume', 
                     'greek_legal_code_chapter', 'greek_legal_code_subject', 
                     'online_terms_of_service_unfairness_levels', 'online_terms_of_service_clause_topics', 
                     'covid19_emergency_event', 'lener_br', 'legalnero', 
                     'greek_legal_ner', 'mapa_coarse', 'mapa_fine'] #'multi_eurlex_level_1'


# In[6]:


def split_into_languages(dataset):
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

def generate_historgram(dataset_name, dataframe,percentile=False,language=None):
    
    if language is None:
        all_data_as_df_filtered = dataframe
    else:
        all_data_as_df_filtered = dataframe[dataframe.language==language]


    plt.figure(figsize=(20,10))
    
    
    plots = list()
    labels = list()
    for lmt in set(models_to_be_used):
        
        if percentile==False:
            bins = np.linspace(0, max(list(all_data_as_df_filtered[lmt])),round(max(list(all_data_as_df_filtered[lmt]))/100))
        
        elif percentile ==True:
            bins = np.linspace(0, np.percentile(sorted(all_data_as_df_filtered[lmt].tolist()), 98),round(max(list(all_data_as_df_filtered[lmt]))/100))
        
        plots.append(list(all_data_as_df_filtered[lmt]))
        labels.append(lmt)
    
    plt.hist(plots, bins, alpha=0.5, label=labels)
    
    
    if percentile==False:
        if dataset_name=='multi_eurlex_level_1':   
            plt.xlim(0, 30000)
        elif dataset_name.startswith('greek_legal_code'):
            plt.xlim(0, 20000)
        

    
    plt.legend(loc='upper right')

    
    if language is None:
        plt.xlabel('Length of input', fontsize=16)
    else:
        plt.xlabel('Length of input for language '+language, fontsize=16)
    plt.ylabel('Frequency of length value', fontsize=16)
    plt.title(dataset_name, fontsize=16)
    
    if language is None:
        plt.savefig('../figures/'+dataset_name+'/histogram_'+'_'.join(dataset_name.split())+'.jpg')
        
    else:
        plt.savefig('../figures/'+dataset_name+'/histogram_'+'_'.join(dataset_name.split())+'__'+language+'.jpg')
        
    
        

def create_histograms(dataset_name, percentile, language='all'):
    
    if percentile == False:
        pathname = '../figures/'
    elif percentile ==True:
        pathname = '../figures_percentile_98/'
    
    if Path(pathname+dataset_name).exists():
        shutil.rmtree(pathname+dataset_name)
        os.mkdir(pathname+dataset_name)
    else:
        os.mkdir(pathname+dataset_name)
        
    
    dataset = load_dataset("joelito/lextreme",dataset_name) 

    all_data_as_df = list()

    for split in ['train','validation','test']:
        df = pd.DataFrame(dataset[split])
        all_data_as_df.append(df)
        

    all_data_as_df = pd.concat(all_data_as_df)
    all_data_as_df = all_data_as_df
    
    if dataset_name.startswith('multi_eurlex'):
        all_data_as_df = split_into_languages(all_data_as_df)
            
    for lmt in models_to_be_used:
        tokenizer = AutoTokenizer.from_pretrained(lmt)
        all_data_as_df[lmt]=all_data_as_df.input.apply(lambda x: get_tokenization_length(tokenizer,x))
    
    if language == 'all':
        
        if 'language' in all_data_as_df.columns.tolist():
            
            for lang in all_data_as_df.language.unique():
            
                generate_historgram(dataset_name,all_data_as_df,lang)
        
        if len(all_data_as_df.language.unique())>1:
            
            generate_historgram(dataset_name,all_data_as_df)
            
    else:
        generate_historgram(dataset_name,all_data_as_df,language)


# In[ ]:


with Pool() as p:
    print(p.map(create_histograms, [(x,False) for x in lextreme_datasets]))
    

#Considering only the data within 98 percentile
with Pool() as p:
    print(p.map(create_histograms, [(x,True) for x in lextreme_datasets]))


# In[ ]:




