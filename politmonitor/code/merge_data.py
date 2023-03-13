import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm
import json as js
from collections import Counter
import fasttext

tqdm.pandas()

# Define all necessary functions and classes

import fasttext


class LanguageIdentification:
    '''From: https://medium.com/@c.chaitanya/language-identification-in-python-using-fasttext-60359dc30ed0'''

    def __init__(self):
        pretrained_lang_model = "../models/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=2)  # returns top 2 matching languages
        return re.sub('__label__', '', predictions[0][0])


language_detector = LanguageIdentification()


def remove_if_not_required_language(text, required_language):
    text = str(text)
    text = re.sub('\n', '', text)

    predicted_language = language_detector.predict_lang(text)

    if predicted_language == required_language:
        return text
    else:
        return ''


def convert_topics_to_list(affair_topic_codes):
    '''Converts the string of semicolon-seperated topic ids into a list'''

    topic_list = str(affair_topic_codes).split(';')
    topic_list = [x for x in topic_list if bool(re.search('\w', x))]
    return topic_list


def strip_html_tags(input_text):
    if type(input_text) == str:
        soup = BeautifulSoup(input_text)
        return soup.text
    else:
        return input_text


# Creating json with id to label and label to id

label_df = pd.read_excel('../data/Data/topics_politmonitor.xlsx')
label_df['keyword_id'] = label_df.keyword_id.apply(lambda x: int(x))
label2id = dict()
id2label = dict()
for record in label_df.to_dict(orient="record"):
    _id = int(record['keyword_id'])
    label = record['keyword_de']
    label2id[label] = _id
    id2label[_id] = label

with open('../utils/id2label.json', 'w') as f:
    js.dump(id2label, f, indent=2, ensure_ascii=False)

with open('../utils/label2id.json', 'w') as f:
    js.dump(label2id, f, indent=2, ensure_ascii=False)

affairs = pd.read_excel('../data/Data/affairs.xlsx')

# Filtering out the following cases
# topics that are nan
# cases where topic id is 1696==Diverses and topic is other

affairs = affairs[affairs.affair_topic_codes.isnull() == False]
affairs = affairs[affairs.affair_topic_codes.isin(['1696', 'other']) == False]
affairs["affair_topic_codes"] = affairs.affair_topic_codes.progress_apply(convert_topics_to_list)

# affairs = affairs[(affairs.affair_scope.isin(['zh','ch']))] # We will keep all kantons for now

affairs = affairs[(affairs.affair_topic_codes.apply(lambda x: len(x) > 0))]

# Read other text parts from csv


all_textual_data = list()
for kanton in affairs.affair_scope.unique():

    print('+++ Processing kanton ' + kanton + ' +++')

    file_name = '../data/Data/documents_content/' + kanton + '.csv.gz'
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, compression='gzip', sep=";")
        df['affair_text_scope'] = kanton
        for x in ["text_de", "text_fr", "text_it"]:
            df[x] = df[x].apply(strip_html_tags)
        all_textual_data.append(df)
    else:
        print('The following file does not exists: ', file_name)

all_textual_data_df = pd.concat(all_textual_data)

# Reset the index
all_textual_data_df.reset_index(drop=True, inplace=True)

# See unique values of text_name
text_names_de = set()
text_names_fr = set()
text_names_it = set()

for x in all_textual_data_df.to_dict(orient="records"):
    text_names_de.add(x['text_name_de'])
    text_names_fr.add(x['text_name_fr'])
    text_names_it.add(x['text_name_it'])

# Remove all content where, for example, the field is text_de, but the actual text is not German
for language in ['de', 'fr', 'it']:
    all_textual_data_df['text_'+language]=all_textual_data_df['text_'+language].progress_apply(lambda x: remove_if_not_required_language(x,language))

# Create a column to merge the df created from the csv with the df for the affairs
# The column will be created by concatinating
#     affair_scope with affair_scope with affair_srcid (for affairs)
#     affair_scope with affair_text_scope with affair_text_srcid (for csv data)

column_name_for_merge = 'id_for_merge'
#affairs[column_name_for_merge] = affairs['affair_scope']+'_'+affairs['affair_srcid'].apply(lambda x: str(x))
#all_textual_data_df[column_name_for_merge] = all_textual_data_df['affair_text_scope']+'_'+all_textual_data_df['affair_text_srcid'].apply(lambda x: str(x))

affairs[column_name_for_merge] = affairs['affair_srcid']
all_textual_data_df[column_name_for_merge] = all_textual_data_df['affair_text_srcid']

# Then, in the dataframe for the csv data, we keep only those columns that do not occur in the affairs dataframe

columns_not_in_affairs = [c for c in all_textual_data_df.columns.tolist() if c not in affairs.columns.tolist()] + [column_name_for_merge]

all_textual_data_df_for_merge = all_textual_data_df[columns_not_in_affairs]

df_merged = all_textual_data_df_for_merge.merge(affairs,on=column_name_for_merge)

df_merged['affair_topic_codes_as_labels'] = df_merged.affair_topic_codes.apply(lambda label_list: [id2label[int(x)] for x in label_list])


# Saving
df_merged.to_json('../data/raw_data_for_training.jsonl', lines=True, orient="records", force_ascii=False)
