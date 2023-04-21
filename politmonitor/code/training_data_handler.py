import json as js
import os
from collections import Counter
from copy import deepcopy
from typing import Union
from collections import defaultdict
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from datasets import Dataset, DatasetDict
from sklearn.utils import shuffle


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

    def process_argument(self, language, default_values):

        if 'all' in language:
            language = default_values
        elif type(language) == str and language != 'all':
            language = language.split(',')
            language = [l.strip() for l in language]
        return language

    def get_training_data(self,
                          language: Union[list, str],
                          affair_attachment_category: Union[list, str],
                          affair_text_scope: Union[list, str],
                          same_items_for_each_language=True,
                          test_size=0.4,
                          running_mode="default",
                          merge_texts=False):

        training_data_df = self.filter_training_data(language=language,
                                                     affair_text_scope=affair_text_scope,
                                                     affair_attachment_category=affair_attachment_category,
                                                     same_items_for_each_language=same_items_for_each_language,
                                                     merge_texts=merge_texts
                                                     )

        training_data_df = self.create_df_for_split(training_data_df)

        training_data_df = self.create_split(training_data_df, test_size=test_size)

        if running_mode in {'experimental', 'debug'}:
            # We keep the all validation and test examples
            indices_to_keep = training_data_df[
                training_data_df.split.isin(['validation', 'test'])].index.tolist()  # ['validation', 'test']
            counter_dict = defaultdict(list)
            # We just filter the training examples
            for i, r in training_data_df[training_data_df.split == 'train'].iterrows():
                language = training_data_df.at[i, 'language']
                '''_key = self.training_data_df.at[i, 'one_hot_affair_topic_codes']
                _key = [str(x) for x in _key]
                _key = '_'.join(_key)'''
                affair_topic_codes_as_labels = training_data_df.at[i, 'affair_topic_codes_as_labels']
                affair_attachment_category = training_data_df.at[i, 'affair_attachment_category']
                labels_for_checking = list(zip(affair_attachment_category, affair_topic_codes_as_labels))
                labels_for_checking = ['_'.join(list(x)) for x in labels_for_checking]
                for _key in labels_for_checking:
                    _key = _key + '_' + language
                    if _key not in counter_dict.keys():
                        counter_dict[_key].append(i)
                        indices_to_keep.append(i)
                    else:
                        if running_mode == 'experimental':
                            maximum = 40
                        elif running_mode == 'debug':
                            maximum = 6
                        if len(counter_dict[_key]) < maximum:
                            counter_dict[_key].append(i)
                            indices_to_keep.append(i)

            training_data_df = training_data_df.iloc[indices_to_keep]

        # Shuffle the training data randomly
        training_data_df = shuffle(training_data_df, random_state=42)

        self.training_data_df = training_data_df

        self.training_data = self.convert_dataframe_to_dataset(self.training_data_df,
                                                               'affair_topic_codes_as_labels')

    def convert_dataframe_to_dataset(self, dataframe, label_column):

        dataset = DatasetDict()
        for split in ['train', 'validation', 'test']:
            data = list()
            df_split = dataframe[dataframe.split == split]
            df_split = df_split.applymap(lambda x: convert_to_string(x))
            for r in df_split.to_dict(orient="records"):
                _input = r['text']
                label = r[label_column]
                r['input'] = _input
                r['label'] = label
                data.append(r)

            data = Dataset.from_list(data)
            dataset[split] = data

        # dataset = Dataset.from_dict(dataset)

        return dataset

    def filter_training_data(self,
                             merge_texts: bool,
                             language: Union[list, str],
                             affair_attachment_category: Union[list, str],
                             affair_text_scope: Union[list, str],
                             same_items_for_each_language=True
                             ):

        training_data = deepcopy(self.raw_data_for_training)

        if affair_text_scope == 'all':
            affair_text_scope = ['all']
        if affair_text_scope != ['all']:
            if type(affair_text_scope) == str:
                affair_text_scope = affair_text_scope.split(',')
            training_data = training_data[training_data.affair_text_scope.isin(affair_text_scope)]

        language = self.process_argument(language, ['de', 'it', 'fr'])
        training_data = training_data[training_data.language.isin(language)]

        affair_attachment_category = self.process_argument(affair_attachment_category,
                                                           ['Titel', 'Text 1', 'Text 2', 'Text 3'])
        training_data = training_data[training_data.affair_attachment_category.isin(affair_attachment_category)]

        # TODO: Add a function to remove cases where the translations seems not correct
        training_data = training_data.drop_duplicates(['text', 'title', 'affair_text_srcid', 'language'])

        # same_items_for_each_language: If this is set to True, it will make sure that for each affair_text_srcid we
        # have a version of the text for each chosen language
        if same_items_for_each_language:
            affair_text_srcid_lists = [training_data[training_data.language == lang].affair_text_srcid.tolist() for lang
                                       in language]
            affair_text_srcid_in_all_languages = set.intersection(*map(set, affair_text_srcid_lists))
            training_data = training_data[
                training_data.affair_text_srcid.isin(list(affair_text_srcid_in_all_languages))]

        # We decided that we will note merge texts that belong to the same affair_text_srcid
        if merge_texts:
            training_data = self.merge_texts(training_data, ['text'])

        return training_data

    def take_first_if_only_1_item(self, item):
        if type(item) == list:
            item_with_unique_values = list()
            for i in item:
                if i not in item_with_unique_values:
                    item_with_unique_values.append(i)
            if len(item_with_unique_values) == 1:
                return item_with_unique_values[0]
            else:
                return "; ".join(item_with_unique_values)
        else:
            return str(item)


    def merge_texts(self, dataframe, input_columns, key_to_group_by='affair_text_srcid'):

        dataframe_grouped = dataframe.groupby(['language', key_to_group_by], as_index=False).agg(list)

        columns = dataframe.columns.tolist()

        for c in columns:
            if c in input_columns:
                dataframe_grouped[c] = dataframe_grouped[c].apply(lambda items: ' '.join(list(dict.fromkeys(items))))
            else:
                dataframe_grouped[c] = dataframe_grouped[c].apply(lambda x: self.take_first_if_only_1_item(x))

        # dataframe_grouped = dataframe_grouped.applymap(lambda x: self.take_first_if_only_1_item(x))

        dataframe_grouped = dataframe_grouped.fillna('')
        return dataframe_grouped

    def create_df_for_split(self, initial_df):

        # initial_df = initial_df.drop_duplicates(colum_for_duplicate_removal)

        data_for_split = list()

        for item in initial_df.to_dict(orient='records'):
            # item = dict()
            # affair_text_srcid = r['affair_text_srcid']
            affair_topic_codes_as_labels = item['affair_topic_codes_as_labels']
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
                del item_copy['text_' + lang]
                del item_copy['title_' + lang]
                if text:
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
            dataframe_unique_affair_text_srcid.loc[
                (dataframe_unique_affair_text_srcid.index.isin(train_index)), 'split'] = 'train'
            dataframe_unique_affair_text_srcid.loc[
                (dataframe_unique_affair_text_srcid.index.isin(other_index)), 'split'] = 'other'

            dataframe_filtered = dataframe_unique_affair_text_srcid[
                dataframe_unique_affair_text_srcid.split == 'other'].reset_index(drop=False)
            X = dataframe_filtered.affair_text_srcid.values
            y = np.array(dataframe_filtered.one_hot_affair_topic_codes.tolist())

            msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

            for validation_index, test_index in msss.split(X, y):
                validation_affair_text_srcid = dataframe_filtered[
                    dataframe_filtered.index.isin(validation_index)].affair_text_srcid.tolist()
                test_affair_text_srcid = dataframe_filtered[
                    dataframe_filtered.index.isin(test_index)].affair_text_srcid.tolist()
                dataframe_unique_affair_text_srcid.loc[dataframe_unique_affair_text_srcid.affair_text_srcid.isin(
                    validation_affair_text_srcid), 'split'] = 'validation'
                dataframe_unique_affair_text_srcid.loc[
                    dataframe_unique_affair_text_srcid.affair_text_srcid.isin(test_affair_text_srcid), 'split'] = 'test'

        look_up_dict = dict()

        for r in dataframe_unique_affair_text_srcid.to_dict(orient="records"):
            look_up_dict[r["affair_text_srcid"]] = r["split"]

        dataframe['split'] = dataframe.affair_text_srcid.apply(lambda x: look_up_dict[x])

        return dataframe

    def create_barplot(self, df, split='all', title='', return_table=False):
        if split != 'all':
            df = df[df.split == split]
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
