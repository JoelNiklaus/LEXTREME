import json as js
import os
from collections import Counter
from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from datasets import Dataset, DatasetDict


def convert_to_string(item):
    if type(item) == list:
        return item
    else:
        return str(item)


class TrainingDataHandler:

    def __init__(self):
        raw_data_for_training = pd.read_json(self.get_relative_path('../data/raw_data_for_training.jsonl'),
                                             lines=True)
        raw_data_for_training.fillna('', inplace=True)

        self.raw_data_for_training = self.split_into_language(raw_data_for_training)

        with open(self.get_relative_path('../utils/label2id.json')) as f:
            self.label2id = js.load(f)

        with open(self.get_relative_path('../utils/id2label.json')) as f:
            self.id2label = js.load(f)

    def get_relative_path(self, path):
        return os.path.join(os.path.dirname(__file__), path)

    def process_language(self, language):

        if 'all' in language:
            language = ['de', 'it', 'fr']
        elif type(language)==str and language!='all':
            language = language.split(',')
            language = [l.strip() for l in language]
     

        return language

    def get_training_data(self, language: Literal['de', 'it', 'fr', 'all'],
                            title: bool = True,
                            text: bool = True,
                            affair_text_scope=['all']):

        self.training_data_df = self.filter_training_data(language=language, 
                                                          title=title,
                                                          text=text,
                                                          affair_text_scope=affair_text_scope
                                                          )
        self.training_data_df = self.create_df_for_split(self.training_data_df)
        self.training_data_df = self.create_split(self.training_data_df)

        language = self.process_language(language)

        inputs = list()

        if title:
            inputs.append('title')

        if text:
            inputs.append('text')

        self.training_data = self.convert_dataframe_to_dataset(self.training_data_df, inputs,
                                                               'affair_topic_codes_as_labels')

    def convert_dataframe_to_dataset(self, dataframe, input_columns, label_column):

        dataset = DatasetDict()
        for split in ['train', 'validation', 'test']:
            data = list()
            df_split = dataframe[dataframe.split == split]
            df_split = df_split.applymap(lambda x: convert_to_string(x))
            for r in df_split.to_dict(orient="records"):
                _input = ' '.join([value for key, value in r.items() if key in input_columns])
                label = r[label_column]
                r['input'] = _input
                r['label'] = label
                data.append(r)

            data = Dataset.from_list(data)
            dataset[split] = data

        # dataset = Dataset.from_dict(dataset)

        return dataset

    def filter_training_data(self, language: Literal['de', 'it', 'fr', 'all'],
                             title: bool,
                             text: bool,
                             affair_text_scope=['all'],
                             same_items_for_each_language=True
                             ):
        
        training_data = deepcopy(self.raw_data_for_training)

        if affair_text_scope == 'all':
            affair_text_scope = ['all']
        if affair_text_scope != ['all']:
            if type(affair_text_scope) == str:
                affair_text_scope = affair_text_scope.split(',')
            training_data = training_data[training_data.affair_text_scope.isin(affair_text_scope)]

        language = self.process_language(language)

        training_data = training_data[training_data.language.isin(language)]

        inputs = list()

        if title:
            inputs.append('title')
            training_data = training_data[training_data.title!='']

        if text:
            inputs.append('text')
            training_data = training_data[training_data.text!='']

        # TODO: Add a function to remove cases where the translations seems not correct

        # If someone chooses only title as input, that only unique entries for tile + affair_text_srcid will remain
        # If someone chooses only title and text as input, that only unique entries for tile + text + affair_text_srcid will remain
        training_data = training_data.drop_duplicates(inputs + ['affair_text_srcid','language'])

        # same_items_for_each_language: If this is set to True, it will make sure that for each affair_text_srcid we have a version for each chosen language
        if same_items_for_each_language:
            affair_text_srcid_lists = [training_data[training_data.language==lang].affair_text_srcid.tolist() for lang in language]
            print('affair_text_srcid_lists', sum([len(x) for x in affair_text_srcid_lists]))
            affair_text_srcid_in_all_languages = set.intersection(*map(set,affair_text_srcid_lists))
            print('affair_text_srcid_in_all_languages', len(affair_text_srcid_in_all_languages))

            training_data = training_data[training_data.affair_text_srcid.isin(list(affair_text_srcid_in_all_languages))]

        return training_data

    def create_df_for_split(self, initial_df):

        # initial_df = initial_df.drop_duplicates(colum_for_duplicate_removal)

        data_for_split = list()

        for item in initial_df.to_dict(orient='records'):
            # item = dict()
            # affair_text_srcid = r['affair_text_srcid']
            affair_topic_codes_as_labels = item['affair_topic_codes']
            # affair_topic_codes = r['affair_topic_codes']
            # item['affair_text_srcid'] = affair_text_srcid
            # item['affair_topic_codes_as_labels'] = affair_topic_codes_as_labels
            # item['affair_topic_codes'] = affair_topic_codes

            one_hot_affair_topic_codes = []
            for label in sorted(list(self.label2id.keys())):
                if label in affair_topic_codes_as_labels:
                    item[label] = 1
                    one_hot_affair_topic_codes.append(1)
                else:
                    item[label] = 0
                    one_hot_affair_topic_codes.append(0)

            item['one_hot_affair_topic_codes'] = one_hot_affair_topic_codes

            data_for_split.append(item)

        data_for_split_df = pd.DataFrame(data_for_split)

        data_for_split_df = data_for_split_df.reset_index(drop=True)

        return data_for_split_df

    def split_into_language(self, dataset_df):

        dataset_new = list()

        for item in dataset_df.to_dict(orient='records'):
            for lang in ['de', 'fr', 'it']:
                item_copy = deepcopy(item)
                text = item['text_' + lang]
                title = item['title_' + lang]
                if title == text: # In these cases, the text field is just a copy of the title field, therefore we set text to empty string
                    text= ""
                del item_copy['text_' + lang]
                del item_copy['title_' + lang]
                item_copy['text'] = text
                item_copy['title'] = title
                item_copy['language'] = lang
                dataset_new.append(item_copy)

        dataset_new = pd.DataFrame(dataset_new)

        return dataset_new

    def create_split(self, dataframe, test_size=0.4):
        if 'split' not in dataframe.columns:
            dataframe['split'] = ''

        dataframe_unique_affair_text_srcid = dataframe.drop_duplicates('affair_text_srcid').reset_index(drop=False)

        X = dataframe_unique_affair_text_srcid.affair_text_srcid.values
        y = np.array(dataframe_unique_affair_text_srcid.one_hot_affair_topic_codes.tolist())

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)

        for train_index, other_index in msss.split(X, y):
            dataframe_unique_affair_text_srcid.loc[(dataframe_unique_affair_text_srcid.index.isin(train_index)), 'split'] = 'train'
            dataframe_unique_affair_text_srcid.loc[(dataframe_unique_affair_text_srcid.index.isin(other_index)), 'split'] = 'other'

            dataframe_filtered = dataframe_unique_affair_text_srcid[dataframe_unique_affair_text_srcid.split == 'other'].reset_index(drop=False)
            X = dataframe_filtered.affair_text_srcid.values
            y = np.array(dataframe_filtered.one_hot_affair_topic_codes.tolist())

            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

            for validation_index, test_index in msss.split(X, y):
                validation_affair_text_srcid = dataframe_filtered[dataframe_filtered.index.isin(validation_index)].affair_text_srcid.tolist()
                test_affair_text_srcid = dataframe_filtered[dataframe_filtered.index.isin(test_index)].affair_text_srcid.tolist()
                dataframe_unique_affair_text_srcid.loc[dataframe_unique_affair_text_srcid.affair_text_srcid.isin(validation_affair_text_srcid), 'split'] = 'validation'
                dataframe_unique_affair_text_srcid.loc[dataframe_unique_affair_text_srcid.affair_text_srcid.isin(test_affair_text_srcid), 'split'] = 'test'
                

        look_up_dict = dict()

        for r in dataframe_unique_affair_text_srcid.to_dict(orient="records"):
            look_up_dict[r["affair_text_srcid"]]=r["split"]

        dataframe['split'] = dataframe.affair_text_srcid.apply(lambda x: look_up_dict[x])

        return dataframe

    def create_barplot(self, df, title='', return_table=False):
        all_labels = list()
        for label_list in df.affair_topic_codes_as_labels:
            for label in label_list:
                all_labels.append(label)

        labels_counted = dict(Counter(all_labels))
        labels_counted = dict(sorted([x for x in labels_counted.items()], key=lambda x: x[1], reverse=True))
        # for label, count in labels_counted.items():
        # print(label,': ',count, end=' ; ')
        plot_data = pd.DataFrame([labels_counted])
        plot_data = plot_data.sort_values(by=0, ascending=False, axis=1)

        if return_table:
            return plot_data
        else:
            ax = plot_data.plot.bar(figsize=(20, 15), width=2.2, title=title)
            for container in ax.containers:
                ax.bar_label(container)
