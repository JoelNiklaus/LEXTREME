import pandas as pd
import os
from collections import defaultdict


def get_relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)


class LabelHandler:

    def __init__(self):
        labels = pd.read_excel(get_relative_path('../data/Data/topics_curia_vista.xlsx'))
        self.labels = labels

    def get_label_translator(self):
        label_translator = defaultdict(dict)
        for item in self.labels.to_dict(orient='records'):
            code = item['code']
            language = item['language'].lower()
            label_name = item['name']
            label_translator[code][language] = label_name

        label_translator = dict(label_translator)

        return label_translator

    def get_all_labels_per_language(self, language):
        label_list = self.labels[self.labels.language.apply(lambda x: str(x).lower() == language)].name.unique()
        label_list = sorted(list(label_list))
        return label_list
